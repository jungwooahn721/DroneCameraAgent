import argparse
import json
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm

def load_metric_module(metric_name):
    # Map metric names to filenames
    # e.g. 'stddev' -> 'pipelines/infer_stddev.py'
    module_name = f"infer_{metric_name}"
    file_path = Path(__file__).parent / "pipelines" / f"{module_name}.py"
    
    if not file_path.exists():
        print(f"Error: Pipeline for metric '{metric_name}' not found at {file_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, "infer"):
        print(f"Error: Module {module_name} does not have an 'infer' function.")
        sys.exit(1)
        
    return module

def main():
    parser = argparse.ArgumentParser(description="Score rendered images using various metrics.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the run directory (containing images/ and annotations.json)")
    parser.add_argument("--metric", type=str, required=True, help="Metric to use (e.g., stddev, laplacian, brightness)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist.")
        sys.exit(1)
        
    annotations_path = output_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found. Cannot update scores.")
        sys.exit(1)
        
    # Load metric module
    metric_module = load_metric_module(args.metric)
    
    # Load annotations
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
        
    print(f"Scoring {len(annotations)} images with metric '{args.metric}'...")
    
    # Process images
    for item in tqdm(annotations):
        # Image path is relative in annotations.json (e.g., "images/img_0000.png")
        # We need to construct the full path
        rel_path = item.get("image")
        if not rel_path:
            continue
            
        image_path = output_dir / rel_path
        
        if not image_path.exists():
            print(f"Warning: Image {image_path} not found.")
            score = 0.0
        else:
            score = metric_module.infer(image_path)
            
        # Store score in the annotation item
        # Key format: "score_<metric>"
        item[f"score_{args.metric}"] = score
        
    # Save updated annotations
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)
        
    print(f"Updated {annotations_path} with scores.")

if __name__ == "__main__":
    main()

"""Example Usage: stddev laplacian brightness
CUDA_VISIBLE_DEVICES=6 python score_images.py --output_dir outputs/Koky_LuxuryHouse_0_251222_105729 --metric qalign
"""