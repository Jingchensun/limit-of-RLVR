import os
import gdown
import shutil

# Define the Google Drive file IDs for the JSON files
FILE_IDS = {
    "test_livecodebench.json": "1B0sotl48BLd4gqlitL5HVJf1cy3RxpEV",
}

# Default values
HOME="YOU/NEED/TO/FIll/IT"

# Define the destination paths
DEST_PATHS = {
    "test_livecodebench.json": HOME + "/data/test_livecodebench.json",
}

# Create the necessary directories
for path in DEST_PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Download and move files
for filename, file_id in FILE_IDS.items():
    temp_path = f"./{filename}"  # Download location
    dest_path = DEST_PATHS[filename]

    # Download the file
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)

    # Move to the correct location
    print(f"Moving {filename} to {dest_path}...")
    shutil.move(temp_path, dest_path)

print("All files downloaded and moved successfully.")