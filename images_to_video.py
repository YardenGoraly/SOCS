import cv2
import os

print('converting to video')

# Directory containing the PNG images
image_dir = 'data_raw/train/video_0'

# Get the list of PNG files in the directory
png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Sort the PNG files based on their names
png_files.sort()

# Read the first image to get dimensions
first_image = cv2.imread(os.path.join(image_dir, png_files[0]))
height, width, _ = first_image.shape

# Define the output video file name
output_video = 'output.mp4'

# Initialize VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Iterate over PNG files and write to video
for png_file in png_files:
    image_path = os.path.join(image_dir, png_file)
    frame = cv2.imread(image_path)
    out.write(frame)

# Release VideoWriter
out.release()

print("Video created successfully!")