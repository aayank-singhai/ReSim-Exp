import os
import subprocess

# Define your input and output directories
INPUT_BASE_DIR = "/home/aic7kor/Desktop/Bosch/ReSim/dataset/video"
OUTPUT_BASE_DIR = "/home/aic7kor/Desktop/Bosch/ReSim/dataset/images"

def extract_frames():
    # 1. Get just the video filename from the user
    video_filename = input("Enter the video filename (e.g., 1ca3b511...mp4): ").strip()

    # 2. Construct the full path to the video file
    video_path = os.path.join(INPUT_BASE_DIR, video_filename)

    # Check if the file actually exists in the video folder before proceeding
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Could not find '{video_filename}' inside {INPUT_BASE_DIR}")
        return

    # 3. Strip the extension (.mp4) to get the clean folder name for the images
    folder_name, _ = os.path.splitext(video_filename)

    # 4. Create the full path for the new directory in the images folder
    output_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)

    # Create the directory (exist_ok=True ensures it doesn't crash if the folder is already there)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory ready: {output_dir}")

    # 5. Define the output pattern for ffmpeg
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    # 6. Construct the ffmpeg command using the precise input path
    command = [
        "ffmpeg",
        "-i", video_path,  
        "-vf", "fps=8",
        output_pattern
    ]

    print("🎬 Extracting frames... Please wait.\n")

    # 7. Run the command safely
    try:
        subprocess.run(command, check=True)
        print("\n✅ Frame extraction completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred while running ffmpeg: {e}")
    except FileNotFoundError:
        print("\n❌ Error: 'ffmpeg' is not installed or not found in your system's PATH.")

if __name__ == "__main__":
    extract_frames()