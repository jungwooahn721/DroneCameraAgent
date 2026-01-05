import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-384"

# Positive prompt: What we WANT to see
POSITIVE_PROMPT = "a photo of a man"

# Negative prompts: What we DO NOT want to see (failure modes)
NEGATIVE_PROMPTS = [
    "a black image",
    "nothing",
    "solid color",
    "noise",
    "extreme close-up of texture",
    "blurry image"
]

_processor = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _processor, _model
    if _model is not None:
        return

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    _model.to(_device)
    _model.eval()


def infer(image_path):
    _load_model()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return 0.0

    # Compare "Positive" vs "All Negatives"
    texts = [POSITIVE_PROMPT] + NEGATIVE_PROMPTS
    
    inputs = _processor(
        text=texts, images=image, padding=True, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        
        # Calculate probabilities using Softmax across all prompts
        logits_per_image = outputs.logits_per_image  # shape: [1, num_prompts]
        probs = logits_per_image.softmax(dim=1)      # shape: [1, num_prompts]
        
        # The score is the probability of the POSITIVE prompt
        # This will be between 0.0 and 1.0
        score = probs[0][0].item()

    return float(score)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate SigLIP2 score of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    print(infer(args.image_path))