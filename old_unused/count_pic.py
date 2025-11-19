import argparse, json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    if not labels:
        raise ValueError("labels.txt is empty")
    return labels

def device_auto() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def parse_args():
    p = argparse.ArgumentParser("detect_and_count")
    p.add_argument("image", help="Path to image")
    p.add_argument("--labels", default="data/labels.txt")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--focus", help="Count only this label (e.g., 'banana')")
    p.add_argument("--json", action="store_true", help="Output counts as JSON")
    return p.parse_args()

def main():
    args = parse_args()
    labels = load_labels(args.labels)
    prompts = [[f"a photo of a {c}" for c in labels]]

    dev = device_auto()
    model_id = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(dev).eval()

    img_path = Path(args.image)
    image = Image.open(img_path).convert("RGB")

    inputs = proc(text=prompts, images=image, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(dev)
    res = proc.post_process_object_detection(
        out, target_sizes=target_sizes, threshold=args.threshold
    )[0]

    # Tally by label
    counts = Counter()
    for score, lid in zip(res["scores"].tolist(), res["labels"].tolist()):
        lbl = labels[lid]
        counts[lbl] += 1

    if args.focus:
        n = counts.get(args.focus, 0)
        suffix = "" if args.focus.endswith("s") else "s"
        print(f"{args.focus}{suffix}: {n}")
    else:
        if args.json:
            print(json.dumps(counts, ensure_ascii=False))
        else:
            for lbl, n in counts.items():
                suff = "" if lbl.endswith("s") else "s"
                print(f"{lbl}{suff}: {n}")

if __name__ == "__main__":
    main()