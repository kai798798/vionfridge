from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_LABELS_PATH = "data/labels.txt"
DEFAULT_PROMPT_TEMPLATE = "a studio photo of a fresh {label} on a white background"

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

@dataclass
class Prediction:
    label: str
    score: float

@dataclass
class Result:
    image: str
    predictions: List[Prediction]


def discover_device(user_choice: str | None = None) -> str:
    if user_choice:
        return user_choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_labels(
    labels_path: str | Path | None, labels_inline: Sequence[str] | None
) -> List[str]:
    if labels_inline and any(lbl.strip() for lbl in labels_inline):
        labels = [lbl.strip() for lbl in labels_inline if lbl.strip()]
    else:
        path = Path(labels_path or DEFAULT_LABELS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {path}. Provide --labels or --labels-inline."
            )
        with path.open("r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
    if not labels:
        raise ValueError("No labels provided. labels list is empty.")
    return labels


def expand_image_paths(
    inputs: Sequence[str], recursive: bool
) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower() in VALID_IMG_EXTS:
                paths.append(p)
        elif p.is_dir():
            globber = p.rglob("*") if recursive else p.glob("*")
            for q in globber:
                if q.is_file() and q.suffix.lower() in VALID_IMG_EXTS:
                    paths.append(q)
        else:
            for q in Path().glob(inp):
                if q.is_file() and q.suffix.lower() in VALID_IMG_EXTS:
                    paths.append(q)
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    if not out:
        raise FileNotFoundError("No image files found in provided inputs.")
    return out

def build_prompts(labels: Sequence[str], template: str) -> List[str]:
    return [template.format(label=lbl) for lbl in labels]

def load_clip(model_id: str, device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def classify_one(
    image_path: Path,
    labels: Sequence[str],
    prompts: Sequence[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    topk: int,
) -> Result:
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        return Result(
            image=str(image_path),
            predictions=[Prediction(label=f"__error__: {type(e).__name__}", score=0.0)],
        )

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=-1).squeeze(0)

    k = max(1, min(topk, len(labels)))
    vals, idxs = torch.topk(probs, k)
    preds = [Prediction(label=labels[i], score=float(vals[j])) for j, i in enumerate(idxs.tolist())]
    return Result(image=str(image_path), predictions=preds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="classify_food.py",
        description="Zero-shot food classifier using CLIP (local, free, CPU-friendly).",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s), folder(s), or glob(s).",
    )
    p.add_argument(
        "--labels",
        default=DEFAULT_LABELS_PATH,
        help=f"Path to labels file (default: {DEFAULT_LABELS_PATH}).",
    )
    p.add_argument(
        "--labels-inline",
        help='Comma-separated labels that override --labels (e.g., "banana,cabbage,bok choy").',
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Force a device; defaults to auto-detect (mps>cuda>cpu).",
    )
    p.add_argument(
        "--topk", type=int, default=1, help="How many predictions to return (default: 1)."
    )
    p.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help=f'Prompt template with "{{label}}" placeholder (default: "{DEFAULT_PROMPT_TEMPLATE}").',
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when inputs include directories.",
    )
    p.add_argument(
        "--output",
        choices=["json", "text"],
        default=None,
        help="Output format. Default: 'text' if topk==1 else 'json'.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_mode = args.output or ("text" if args.topk == 1 else "json")

    labels_inline = (
        [s for s in args.labels_inline.split(",")] if args.labels_inline else None
    )
    labels = load_labels(args.labels, labels_inline)
    prompts = build_prompts(labels, args.prompt_template)

    device = discover_device(args.device)
    model, processor = load_clip(args.model_id, device)

    img_paths = expand_image_paths(args.inputs, recursive=bool(args.recursive))

    results: List[Result] = []
    for img_path in img_paths:
        res = classify_one(
            image_path=img_path,
            labels=labels,
            prompts=prompts,
            model=model,
            processor=processor,
            device=device,
            topk=int(args.topk),
        )
        results.append(res)

    if output_mode == "json":
        for r in results:
            print(
                json.dumps(
                    {
                        "image": r.image,
                        "predictions": [
                            {"label": p.label, "confidence": round(p.score, 6)}
                            for p in r.predictions
                        ],
                    },
                    ensure_ascii=False,
                )
            )
    else:
        if int(args.topk) == 1:
            for r in results:
                print(r.predictions[0].label if r.predictions else "__no_pred__")
        else:
            for r in results:
                for p in r.predictions:
                    print(f"{r.image}\t{p.label}\t{p.score:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())