import re
import subprocess
from pathlib import Path

# Define input and output directories
input_dir = Path("videos")
output_dir = Path("audios")

# Ensure output directory exists
output_dir.mkdir(exist_ok=True)

# Get all files in input directory
video_files = [f for f in input_dir.iterdir() if f.suffix.lower() == ".mp4"]

if not video_files:
    print("No MP4 files found in 'videos' folder.")
else:
    print(f"Found {len(video_files)} video(s) in '{input_dir}' folder.")

# Process each video file
for video_file in video_files:
    try:
        # Extract file number (before first ' - ')
        file_no = video_file.stem.split(" - ", 1)[0].strip()

        # Take the last segment after the final ' - '
        last_part = video_file.stem.split(" - ")[-1].strip()

        # Remove any leading numbers, colons, punctuation
        topic = re.sub(r'^[\d\s：:.\-]+', '', last_part).strip()

        # Sanitize topic for filename (remove invalid characters)
        safe_topic = re.sub(r'[\\/*?:"<>|]', '', topic)

        # Build output filename and path
        output_file = output_dir / f"{file_no}_{safe_topic}.mp3"

        # Run ffmpeg to convert
        print(f"Converting: {video_file.name} → {output_file.name}")
        subprocess.run(
            ["ffmpeg", "-i", str(video_file), str(output_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        print(f"Saved: {output_file}")

    except subprocess.CalledProcessError:
        print(f"Failed to convert: {video_file.name}")
    except Exception as e:
        print(f"Error processing {video_file.name}: {e}")

print("\nConversion complete! All audio files saved in 'audios/' folder.")
