import cv2
import numpy as np

"""
Calculates the standard deviation of pixel intensities.
- Low value (< 10): Image is likely a solid color (flat), indicating the camera might be inside an object or looking at a featureless surface.
- High value: Image has good contrast.
"""

def infer(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(np.std(img))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate standard deviation of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    print(infer(args.image_path))
