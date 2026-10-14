"""Zip a folder of photos and start a Flux LoRA training on Replicate.

Usage: python train.py photos/ your-username/down-memory-lane [--trigger TOK] [--steps 1000] [--wait]
"""

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import replicate
from dotenv import load_dotenv
from replicate.exceptions import ReplicateError

TRAINER = "ostris/flux-dev-lora-trainer"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def zip_photos(folder):
    photos = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not photos:
        sys.exit(f"No images found in {folder}")
    if len(photos) < 10:
        print(f"Warning: only {len(photos)} photos; 15-20 varied photos give a better likeness.")
    archive = Path(tempfile.gettempdir()) / "down-memory-lane-photos.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for photo in photos:
            # Captions are optional: a photo.txt next to photo.jpg is included if present.
            zf.write(photo, photo.name)
            caption = photo.with_suffix(".txt")
            if caption.exists():
                zf.write(caption, caption.name)
    print(f"Zipped {len(photos)} photos -> {archive}")
    return archive


def ensure_destination(destination):
    owner, name = destination.split("/", 1)
    try:
        replicate.models.get(destination)
        print(f"Using existing model {destination}")
    except ReplicateError:
        replicate.models.create(owner=owner, name=name, visibility="private", hardware="gpu-h100",
                                description="Flux LoRA for the Down Memory Lane Slack bot")
        print(f"Created private model {destination}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", help="folder of photos of you, alone in frame")
    parser.add_argument("destination", help="owner/model-name to create on Replicate")
    parser.add_argument("--trigger", default="TOK", help="unique word that will stand for you in prompts")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--wait", action="store_true", help="poll until training finishes")
    args = parser.parse_args()

    archive = zip_photos(args.photos)
    ensure_destination(args.destination)

    trainer_version = replicate.models.get(TRAINER).latest_version.id
    with open(archive, "rb") as images:
        training = replicate.trainings.create(
            model=TRAINER,
            version=trainer_version,
            input={
                "input_images": images,
                "trigger_word": args.trigger,
                "steps": args.steps,
                "lora_rank": 16,
                "autocaption": True,
            },
            destination=args.destination,
        )
    print(f"Training started: https://replicate.com/p/{training.id}  (status: {training.status})")

    if not args.wait:
        print("Re-run with --wait, or watch progress at the URL above.")
        return

    while training.status not in ("succeeded", "failed", "canceled"):
        time.sleep(30)
        training.reload()
        print(f"  {time.strftime('%H:%M:%S')} status: {training.status}")

    if training.status == "succeeded":
        print(f"Done. Put this in .env:\nREPLICATE_MODEL={training.output['version']}")
        print(f"TRIGGER_WORD={args.trigger}")
    else:
        print(f"Training {training.status}: {training.error}")


if __name__ == "__main__":
    main()
