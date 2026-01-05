import cv2
import numpy as np

"""
Calculates the mean brightness of the image (0-255).
- Near 0: Image is too dark (underexposed or blocked).
- Near 255: Image is too bright (overexposed).
"""

def infer(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(np.mean(img))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate brightness of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    print(infer(args.image_path))
