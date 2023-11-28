import argparse
import os
import random
import json

import numpy as np
import cv2
from ultralytics import YOLO

def arg_parse():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--input_videos', type=str, help='Path to the txt file containing video paths')
    parser.add_argument('--selfie_sticks', type=str, help='Path to the folder containing selfie sticks')
    parser.add_argument('--sampling_rate', default=0.1, type=float, help='Percentage of frames to select')
    parser.add_argument('--stick_probability', default=0.5, type=float, help='Probability of having a selfie stick in a skeleton')
    parser.add_argument('--angle', type=float, default=0, help='Maximum absolute deviation angle for selfie stick tilt')
    parser.add_argument('--max_compression', default=0, type=float, help='Maximum compression for perspective distortion')
    parser.add_argument('--format', type=str, default='yolov8', help='Format for labelling (e.g., "yolov8")')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu", "cuda")')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--buffer_size', type=int, default=128, help='Size of the buffer for parallel processing')
    
    args = parser.parse_args()
    return args

def create_folders(args):
    # Create the output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create the images and labels folders if they do not exist
    os.makedirs(os.path.join(args.output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "labels"), exist_ok=True)

def process_frames(input_videos, sampling_rate, buffer_size, model, device, selfie_stick_images, selfie_sticks_data, stick_probability, angle, max_compression, output_folder):
    # Check if input_videos is a valid and existing .txt file
    if not os.path.isfile(input_videos):
        raise FileNotFoundError(f"{input_videos} is not a valid file.")

    # Read the video filenames from the input_videos file
    video_filenames = []
    with open(input_videos, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith(('.mp4', '.avi', '.mov')):
                video_filenames.append(line)

    # Check if sampling_rate is a float between 0 and 1
    if not 0 <= sampling_rate <= 1:
        raise ValueError("sampling_rate should be a float between 0 and 1 (inclusive).")

    selected_frames = []

    incremental_id = 0

    total_videos = len(video_filenames)
    video_count = 0
    for video_filename in video_filenames:
        video_count += 1
        # Open the video using cv2
        video = cv2.VideoCapture(video_filename)

        # Read frames from the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        while not frame_count >= total_frames:
            frame_count += 1

            # Print the progress
            print(f"Processing video {video_count}/{total_videos}; frame ({frame_count}/{total_frames})", end="\r", flush=True)
            ret, frame = video.read()

            # Check if the frame is read successfully
            if not ret:
                break

            # Randomly decide whether to include the frame based on sampling_rate
            if random.random() <= sampling_rate:
                # Convert the image to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Check if people are present in the frame using the people_present function
                if people_present(frame, model, device):
                    selected_frames.append(frame)

            if len(selected_frames) >= buffer_size or frame_count >= total_frames:
                incremental_id = process_batch(selected_frames, model, device, selfie_stick_images, selfie_sticks_data, stick_probability, angle, max_compression, output_folder, incremental_id)
                selected_frames = []            

        # Release the video capture object
        video.release()

    return selected_frames


def process_batch(frames, model, device, selfie_stick_images, selfie_sticks_data, stick_probability, angle, max_compression, output_folder, incremental_id):
    # Add the selfie sticks to the frames
    return add_selfie_stick(frames, selfie_stick_images, selfie_sticks_data, model, device, stick_probability, angle, max_compression, output_folder, incremental_id)


def people_present(image, model, device):
    results = model(image, device=device, verbose=False)
    for _ in results[0].boxes:
        return True
        
    return False


def get_selfie_sticks(selfie_sticks):
    # Check if selfie_sticks is a valid and existing folder
    if not os.path.isdir(selfie_sticks):
        raise NotADirectoryError(f"{selfie_sticks} is not a valid folder.")

    # Read the file containing the selfie stick data (it is a json file)
    with open(os.path.join(selfie_sticks, 'data.json'), 'r') as file:
        selfie_sticks_data = json.load(file)

    # Get the filenames of all the selfie sticks
    selfie_stick_images = {}
    for filename in (stick["filename"] for stick in selfie_sticks_data):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image using cv2
            image = cv2.imread(os.path.join(selfie_sticks, filename), cv2.IMREAD_UNCHANGED)

            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            # Append the image to selfie_stick_images
            selfie_stick_images[filename] = image

    return selfie_stick_images, selfie_sticks_data


def add_selfie_stick(frames, selfie_stick_images, selfie_sticks_data, model, device, stick_probability, angle, max_compression, output_folder, incremental_id):
    for frame in frames:
        filename = str(incremental_id).rjust(6, "0")
        
        height_size = frame.shape[0] // 8

        # Extract the skeletons from the frame using the model
        results = model(frame, device=device, verbose=False)[0]

        # new_frame = results.plot()
        new_frame = frame.copy()

        elbows = []
        hands = []

        for person in results.keypoints.xy:
            keypoints = []
            if int(person[7, 0]) != 0 and int(person[7, 1]) != 0 and int(person[9, 0]) != 0 and int(person[9, 1]) != 0:
                keypoints.append((person[7], person[9]))
            if int(person[8, 0]) != 0 and int(person[8, 1]) != 0 and int(person[10, 0]) != 0 and int(person[10, 1]) != 0:
                keypoints.append((person[8], person[10]))

            if len(keypoints) != 0:
                keypoint = random.choice(keypoints)
                elbows.append(keypoint[0])
                hands.append(keypoint[1])

        file = open(os.path.join(output_folder, "labels", f"{filename}.txt"), 'w+')

        # For each skeleton
        for elbow, hand in zip(elbows, hands):
            # Choose whether to add a selfie stick to the skeleton
            if random.random() > stick_probability:
                continue
            
            # Get a random selfie stick data
            stick_data = random.choice(selfie_sticks_data)
            grab_point = np.array(stick_data["grab_point"])

            # Get the image of the selfie stick
            stick_image = selfie_stick_images[stick_data["filename"]].copy()

            # Resize the image to real size of the selfie stick
            width_size = int(height_size * stick_image.shape[1] / stick_image.shape[0])
            grab_point[0] = int(grab_point[0] * width_size / stick_image.shape[1])
            grab_point[1] = int(grab_point[1] * height_size / stick_image.shape[0])
            stick_image = cv2.resize(stick_image, (width_size, height_size))

            # Compress the selfie stick image along the y direction using the cv2 library and without cropping
            rows, cols, channels = stick_image.shape
            compression_factor = 1 - random.uniform(0, max_compression)
            compression_matrix = np.array([[1, 0, 0], [0, compression_factor, 0]], dtype=np.float32)
            rows = int(rows * compression_factor)
            compressed_stick_image = cv2.warpAffine(stick_image, compression_matrix, (cols, rows))
            compressed_grab_point = grab_point.copy()
            compressed_grab_point[1] = compressed_grab_point[1] * compression_factor

            # Randomly change the tonality of the selfie stick image using the cv2 library
            hue_shift = int(random.uniform(0, 180))
            hsv_stick_image = cv2.cvtColor(compressed_stick_image, cv2.COLOR_RGB2HSV)
            hsv_stick_image[:, :, 0] = (hsv_stick_image[:, :, 0] + hue_shift) % 180
            hue_changed_stick_image = np.concatenate([cv2.cvtColor(hsv_stick_image, cv2.COLOR_HSV2RGB), compressed_stick_image[:, :, 3:]], axis=2)

            # Compute the rotation angle
            direction = (hand - elbow).cpu()
            hand_angle = (1 if direction[0] <= 0 else -1) * np.arccos(-direction[1] / np.linalg.norm(direction)).item() * 180 / np.pi
            deviation_angle = random.uniform(-angle, angle)
            rotation_angle = hand_angle + deviation_angle

            # Rotate the selfie stick image using the cv2 library and without cropping
            square_side = int(rows * np.sqrt(2))
            horizontal_offset = (square_side - cols) // 2
            vertical_offset = (square_side - rows) // 2
            zeros = np.zeros((rows, horizontal_offset, channels), dtype=np.uint8)
            square_image = np.concatenate([zeros, hue_changed_stick_image, zeros], axis=1)
            zeros = np.zeros((vertical_offset, square_image.shape[1], channels), dtype=np.uint8)
            square_image = np.concatenate([zeros, square_image, zeros], axis=0)
            square_rows, square_cols, _ = square_image.shape
            rotataion_matrix = cv2.getRotationMatrix2D((square_cols//2, square_rows//2), rotation_angle, 1)
            rotated_stick_image = cv2.warpAffine(square_image, rotataion_matrix, (square_cols, square_rows))

            translated_grab_point = np.array([compressed_grab_point[0] + horizontal_offset, compressed_grab_point[1] + vertical_offset])
            rotated_grab_point = np.matmul(rotataion_matrix, np.array([translated_grab_point[0], translated_grab_point[1], 1])).astype(np.int32)

            # Add the selfie stick to the frame using the cv2 library and without cropping
            hand = hand.to(int)
            xf_min = max(0, hand[0].item() - rotated_grab_point[0])
            xf_max = min(frame.shape[1], hand[0].item() + rotated_stick_image.shape[1] - rotated_grab_point[0])
            yf_min = max(0, hand[1].item() - rotated_grab_point[1])
            yf_max = min(frame.shape[0], hand[1].item() + rotated_stick_image.shape[0] - rotated_grab_point[1])

            xs_min = max(0, rotated_grab_point[0] - hand[0].item())
            xs_max = min(rotated_stick_image.shape[1], rotated_grab_point[0] + frame.shape[1] - hand[0].item())
            ys_min = max(0, rotated_grab_point[1] - hand[1].item())
            ys_max = min(rotated_stick_image.shape[0], rotated_grab_point[1] + frame.shape[0] - hand[1].item())

            new_frame[yf_min:yf_max, xf_min:xf_max][rotated_stick_image[ys_min:ys_max, xs_min:xs_max, 3] != 0] = rotated_stick_image[ys_min:ys_max, xs_min:xs_max, :3][rotated_stick_image[ys_min:ys_max, xs_min:xs_max, 3] != 0]

            # Compute the bounding box of the selfie stick
            stick_coordinates = list(np.where(rotated_stick_image[:, :, 3] != 0))
            stick_coordinates[0] += hand[1].item() - rotated_grab_point[1]
            stick_coordinates[1] += hand[0].item() - rotated_grab_point[0]
            bbox_x_min = np.min(stick_coordinates[1])
            bbox_x_max = np.max(stick_coordinates[1])
            bbox_y_min = np.min(stick_coordinates[0])
            bbox_y_max = np.max(stick_coordinates[0])

            """new_frame = cv2.rectangle(new_frame, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (0, 255, 0, 1), 2)
            
            new_frame = cv2.circle(new_frame, (hand[0].item(), hand[1].item()), 5, (0, 0, 255, 1), 2)
            new_elbow = elbow.to(int)
            new_frame = cv2.circle(new_frame, (new_elbow[0].item(), new_elbow[1].item()), 5, (0, 255, 0, 1), 2)"""

            # Save the label
            
            file.write(f"0 {bbox_x_min / new_frame.shape[1]} {bbox_y_min / new_frame.shape[0]} {(bbox_x_max - bbox_x_min) / new_frame.shape[1]} {(bbox_y_max - bbox_y_min) / new_frame.shape[0]}\n")

        file.close()

        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)

        """cv2.imshow('frame', new_frame)
        cv2.waitKey(1)
        input()"""

        # Save the picture
        cv2.imwrite(os.path.join(output_folder, "images", f"{filename}.png"), new_frame)

        incremental_id += 1

    return incremental_id


def main():
    # Parse the arguments
    args = arg_parse()

    # Create the output folder if it does not exist
    create_folders(args)

    # Initialize the model
    print("Initializing the model...", flush=True)
    model = YOLO('yolov8n-pose.pt')

    # Get the selfie sticks
    print("Getting selfie sticks...", flush=True)
    selfie_stick_images, selfie_sticks_data = get_selfie_sticks(args.selfie_sticks)

    # Extract the frames from the videos
    print("Extracting frames...", flush=True)
    process_frames(args.input_videos, args.sampling_rate, args.buffer_size, model, args.device, selfie_stick_images, selfie_sticks_data, args.stick_probability, args.angle, args.max_compression, args.output_folder)
    print("\nDone", flush=True)


if __name__ == "__main__":
    main()