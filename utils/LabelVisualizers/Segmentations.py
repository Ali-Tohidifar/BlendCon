# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:02:02 2022

@author: Windows
"""

import os
import numpy as np
import pickle
import cv2

RGBforLabel = {
    1: (255, 0, 0),       # Red
    2: (255, 128, 0),     # Orange
    3: (255, 255, 0),     # Yellow
    4: (128, 255, 0),     # Light green-yellow
    5: (0, 255, 0),       # Green
    6: (0, 255, 128),     # Aquamarine
    7: (0, 255, 255),     # Cyan
    8: (0, 128, 255),     # Sky blue
    9: (0, 0, 255),       # Blue
    10: (128, 0, 255),    # Violet
    11: (255, 0, 255),    # Magenta
    12: (255, 0, 128),    # Deep pink
    13: (192, 192, 192),  # Silver
    14: (128, 128, 128),  # Gray
    15: (0, 128, 128),    # Teal
    16: (128, 0, 0),      # Maroon
    17: (128, 128, 0),    # Olive
    18: (0, 0, 128),      # Navy
    19: (139, 69, 19),    # Saddle brown
    20: (244, 164, 96),   # Sandy brown
    21: (250, 128, 114),  # Salmon
    22: (85, 107, 47),    # Dark olive green
    23: (107, 142, 35),   # Olive drab
    24: (199, 21, 133),   # Medium violet red
    25: (70, 130, 180),   # Steel blue
    26: (153, 50, 204),   # Dark orchid
    27: (178, 34, 34),    # Fire brick
    28: (189, 183, 107),  # Dark khaki
    29: (255, 140, 0),    # Dark orange
    30: (72, 209, 204)    # Medium turquoise
}

def visualize_segmentation(raw_data_dir, output_path):
    visualized_data_dir = os.path.join(output_path, "Segmentation")
    
    src_imgs = []
    segment_imgs = []

    # Load and categorize images         
    for root, dirs, files in os.walk(raw_data_dir):
        os.chdir(root)
        
        if 'Semantic Segmentation' in dirs:
            src_imgs = []
            segment_imgs = []
        
        for file in files:
            # Separate segmentations and source images
            if 'Depth Map' in root: 
                continue
            elif 'Semantic Segmentation' in root: 
                if ".jpg" in file: segment_imgs.append(file)
            else:
                if ".jpg" in file: src_imgs.append(file)
        print('Segmentation and Source images are loaded')
        # Creating a visualization directory
        os.makedirs(visualized_data_dir, exist_ok=True)

    # Define the intensity ranges
    ranges = [(i - 4, i + 4) for i in range(1, 245, 10)]
    ranges[0] = (0.5, 4)

    # Iterate over each pair of source and segmentation images
    for src_img_name, seg_img_name in zip(src_imgs, segment_imgs):
        # Load images
        main_img = cv2.imread(os.path.join(raw_data_dir, src_img_name))
        seg_img = cv2.imread(os.path.join(raw_data_dir, 'Semantic Segmentation', seg_img_name), cv2.IMREAD_GRAYSCALE)

        # Create a blank mask with the same dimensions as the source image
        mask_overlay = np.zeros_like(main_img)
        
        for i, (lower, upper) in enumerate(ranges):
            label = i + 1  # Assign a label based on the index of the range; adjust if necessary
            # Create a mask for pixels within the intensity range
            mask = (seg_img >= lower) & (seg_img <= upper)
            
            # Retrieve the corresponding color
            colour = RGBforLabel.get(label)
            if colour is not None:
                # Apply the color mask to the selected pixels
                mask_overlay[mask] = colour
        
        # Blend the color mask with the original image
        alpha = 0.45
        result = cv2.addWeighted(main_img, 1, mask_overlay, alpha, 0)
        
        # Write the result to disk
        filename = os.path.join(visualized_data_dir, src_img_name)
        cv2.imwrite(filename, result)
        print(f'{src_img_name} is done')

def main():
    raw_data_dir = r"C:\Users\Windows\Desktop\data visualization\Dataset\RandomCamera_1_Q4_dronesiviewscom-sep-25-2021-construction-site_V2_Armature"
    output_path = r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6"
    visualize_segmentation(raw_data_dir, output_path)

if __name__ == "__main__":
    main()
