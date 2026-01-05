import cv2

"""
Calculates the variance of the Laplacian (edge detection).
- Low value (< 50): Image is likely blurry or featureless (no edges).
- High value: Image is sharp and textured.
"""

def infer(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Laplacian variance of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    print(infer(args.image_path))
