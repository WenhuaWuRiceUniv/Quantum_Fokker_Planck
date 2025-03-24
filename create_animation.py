import os
import imageio
import argparse
import re  # Regular expressions to extract numbers from filenames

# python create_animation.py <bin_folder> --fps 10
# Set up argument parser
parser = argparse.ArgumentParser(description="Create a video from PNG images.")
parser.add_argument("bin_folder", help="Path to the folder containing PNG images.")
parser.add_argument("--fps", type=int, default=5, help="Frames per second for the video (default: 5).")

# Parse arguments
args = parser.parse_args()

# Define the path to your PNG files
output_folder = args.bin_folder  # Folder to save the output video
bin_folder = args.bin_folder
fps = args.fps  # Frames per second

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all PNG files in the folder
png_files = [f for f in os.listdir(bin_folder) if f.endswith('.png')]

# Function to extract the four-digit number after "snapshot_"
def extract_number(filename):
    match = re.search(r'snapshot_(\d{4})', filename)  # Extract the 4-digit number after "snapshot_"
    return int(match.group(1)) if match else float('inf')  # Return the number or a large number if no match found

# Filter out files that contain a 5-digit number
filtered_files = [f for f in png_files if not re.search(r'snapshot_(\d{5})', f)]

# Sort the remaining files based on the four-digit number extracted from the filename
filtered_files.sort(key=extract_number)

# Select only the first 100 files
filtered_files = filtered_files[:1000]

# Create a list to store the images
images = []

# Load each image and append it to the images list
for file in filtered_files:
    img_path = os.path.join(bin_folder, file)
    img = imageio.imread(img_path)
    images.append(img)

# Define the output file path
output_file = os.path.join(output_folder, 'animation.mp4')

# Save as MP4 (make sure ffmpeg is available)
imageio.mimsave(output_file, images, fps=fps)  # Adjust the FPS as needed

print(f"Video saved to: {output_file}")
