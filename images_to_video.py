import cv2
import os
import argparse
from tqdm import tqdm

def convert_to_videos(in_dir, out_dir, num_seq):
    #in_dir: /data/yardengoraly/object-detection-data/many_items_80000
    # Directory containing the PNG images
    image_dir = 'data_raw/train/video_0'
    videos = sorted(os.listdir(in_dir))

    index = 0
    for video in tqdm(videos[:num_seq]):
        image_dir = os.path.join(in_dir, video)
        # Get the list of PNG files in the directory
        # import pdb; pdb.set_trace()
        png_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]

        # Sort the PNG files based on their names
        png_files.sort()

        # Read the first image to get dimensions
        first_image = cv2.imread(os.path.join(image_dir, png_files[0]))
        height, width, _ = first_image.shape

        # Define the output video file name
        output_video = os.path.join(out_dir, str(index) + '.mp4')

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
        index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val'])
    parser.add_argument('--in_dir', default='data_raw')
    parser.add_argument('--out_dir', default='data_gen')
    parser.add_argument('--num_seq', default = 100)

    args = parser.parse_args()

    in_dir = os.path.join(args.in_dir, args.split)
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'train':
        convert_to_videos(in_dir, out_dir, int(args.num_seq))