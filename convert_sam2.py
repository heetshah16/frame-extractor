import json
import os
import shutil
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Paths
annotations_file = "annotations_new.json"  # <- Update this if different
images_dir = "output/video_frames"                  # <- Directory where your .JPG images are stored
output_dir = "input/debug_frames"

# Initialize model
checkpoint = "sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def infer_sam(image, input_boxes):
    predictor.set_image(image)
    all_masks = []

    for box in input_boxes:
        print(box)
        print((box[0] + box[2]) // 2)
        point = [[(box[0]+box[2])//2, (box[1] +box[3])//2]]
        masks, _, _ = predictor.predict(
            point_coords=point,
            point_labels=[1],
            box=box,
            multimask_output=False,
        )
        # Convert numpy mask to torch tensor
        mask_tensor = torch.from_numpy(masks[0])

        all_masks.append(mask_tensor)

    return torch.stack(all_masks)

# Load new-style annotations
with open(annotations_file, "r") as f:
    annotations = json.load(f)

colors = [
    (255,255,255)
]

os.makedirs(output_dir, exist_ok=True)

# Visualization and processing
for filename, boxes in annotations.items():
    for i in range(len(boxes)):
        frameName = filename+"_"+str(boxes[i]['frame'])+".jpg"
        image_path = os.path.join(images_dir, frameName)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        image_h, image_w, _ = image.shape
        mask_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        input_boxes = np.array([[box["x1"], box["y1"], box["x2"], box["y2"]] for box in boxes[i]['boxes']])
        masks = infer_sam(mask_image, input_boxes)
        mask_bgr = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        for j in range(masks.shape[0]):
            print(masks.shape[0])
            _mask = masks[j]
            print("hi")
            print(_mask.shape)

            mask_bgr[_mask == 1] = colors[0]

        mask_filename = frameName.replace(".jpg", "_mask.png")
        cv2.imwrite(os.path.join(output_dir, mask_filename), mask_bgr)
        shutil.copy(image_path, os.path.join(output_dir, filename))