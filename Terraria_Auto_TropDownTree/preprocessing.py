import cv2
import pandas as pd

def preprocess_video(input_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get the original video's resolution and frame rate
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video.get(cv2.CAP_PROP_FPS)

    # Define the target resolution and frame rate
    target_width = 1280
    target_height = 720
    target_fps = 30

    # Create a VideoWriter object to save the preprocessed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

    while True:
        # Read a frame from the original video
        ret, frame = video.read()

        if not ret:
            break

        # Resize the frame to the target resolution
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Write the resized frame to the output video
        output_video.write(resized_frame)

    # Release the video objects
    video.release()
    output_video.release()

def track_pixels(input_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Create an empty DataFrame to store the pixel data
    df = pd.DataFrame(columns=['name_video'] + [f'pixel_{i}' for i in range(1, 921601)])

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Flatten the frame into a 1D array
        pixels = frame.flatten()

        # Create a dictionary to store the pixel data
        data = {'name_video': input_path}
        for i, pixel in enumerate(pixels):
            data[f'pixel_{i+1}'] = pixel

        # Append the data to the DataFrame
        df = df.append(data, ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)

    # Release the video object
    video.release()