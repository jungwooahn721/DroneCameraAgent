import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import pyiqa

# Global model instance
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    global _MODEL
    if _MODEL is None and pyiqa is not None:
        print(f"Loading Q-Align model on {_DEVICE}...")
        try:
            # Create the metric model
            _MODEL = pyiqa.create_metric('qalign', device=_DEVICE)
        except Exception as e:
            print(f"Failed to create Q-Align metric: {e}")

def infer(image_path):
    """
    Infers the quality score of an image using Q-Align.
    Returns the 'quality' score (technical quality).
    """
    if pyiqa is None:
        return 0.0
        
    _load_model()
    if _MODEL is None:
        return 0.0

    try:
        img = Image.open(image_path).convert('RGB')
        # Convert to tensor [0, 1] and add batch dimension
        input_tensor = to_tensor(img).unsqueeze(0).to(_DEVICE)
        
        with torch.no_grad():
            # task_='quality' for technical quality (distortion, noise, etc.)
            # task_='aesthetic' is also available but we use quality for this purpose
            score = _MODEL(input_tensor, task_='quality')
            
        return float(score.item())
    except Exception as e:
        print(f"Error inferring Q-Align for {image_path}: {e}")
        return 0.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Q-Align quality score of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    print(infer(args.image_path))