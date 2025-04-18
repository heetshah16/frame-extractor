import cv2
import json
from PIL import Image
import imagehash
from pathlib import Path
import os
import shutil

def copy_avi_and_jpg_to_same_folder(src_folder, dest_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Walk through all subdirectories and files
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(('.avi', '.jpg')):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)

                # If a file with the same name already exists, warn and skip (or overwrite if you prefer)
                if os.path.exists(dest_path):
                    print(f"Warning: {dest_path} already exists. Skipping.")
                else:
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied: {src_path} -> {dest_path}")

def extract_and_save_frames(video_path, output_folder, current_frame):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_frames = [f for f in [current_frame - 60, current_frame, current_frame + 60] if 0 <= f < total_frames]

    save_dir = os.path.join(output_folder, "video_frames")
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx in save_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{video_name}_{frame_idx}.jpg"
            frame_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"[INFO] Saved frame {frame_idx} as {frame_path}")
        else:
            print(f"[WARN] Could not read frame {frame_idx}")
def frame_matches_image(frame_img, image_hashes, threshold=10, target_size=(128, 128)):
    """Check if the frame matches any image hash (within threshold)."""
    frame_pil = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)).resize(target_size)
    frame_hash = imagehash.phash(frame_pil)

    for img_name, img_hash in image_hashes.items():
        if abs(frame_hash - img_hash) <= threshold:
            return img_name
    return None


def load_image_hashes(images_dir, target_size=(128, 128)):
    hashes = {}
    for img_path in Path(images_dir).rglob("*.jpg"):
        file_name = img_path.name  # ✅ Just the file name, no folder structure
        with Image.open(img_path) as img:
            img = img.resize(target_size)
            hashes[file_name] = imagehash.phash(img)
    return hashes

def process_videos(folder_path, output_folder_path, output_json_path):
    target_size = (128, 128)
    image_hashes = load_image_hashes(folder_path, target_size)


    result = []
    debug_dir = os.path.join(folder_path, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.lower().endswith('.avi'):
            video_path = os.path.join(folder_path, file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_num = 0

            print(f"Processing video: {file}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save first 10 frames for manual comparison

                matched_img = frame_matches_image(frame, image_hashes, threshold=10, target_size=target_size)

                if matched_img:
                    extract_and_save_frames(video_path,output_folder_path,frame)
                    entry = {
                        "video_name": file,
                        "frame_number": frame_num,
                    }
                    frame_num = min(total_frames - 1, frame_num + 60)
                    result.append(entry)
            cap.release()

    if result:
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nDone. Results written to {output_json_path}")
    else:
        print("\nNo matches found. No JSON file written.")


# Example usage
source_folder = 'dataset'
video_image_folder = "input"
output_json_path = "output/matched_frames.json"
output = "output"

copy_avi_and_jpg_to_same_folder(source_folder, video_image_folder)
process_videos(video_image_folder,output, output_json_path)