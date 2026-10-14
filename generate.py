"""Run one request against the trained model and save the image, to tune prompts without Slack.

Usage: python generate.py "my 5-year-old self on a beach" [--lora-scale 0.8] [--seed 42] [--out outputs/]
"""

import argparse
import os
import time

from dotenv import load_dotenv

from imagegen import build_prompt, generate_image


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", help='e.g. "my 5-year-old self on a beach"')
    parser.add_argument("--lora-scale", type=float, default=None, help="overrides the age-based default (0.65 for ages 1-7, LORA_SCALE otherwise)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--raw", action="store_true", help="send the request text as the prompt unchanged")
    args = parser.parse_args()

    trigger = os.getenv("TRIGGER_WORD", "TOK")
    if args.raw:
        prompt, age = args.request, "raw"
    else:
        prompt, age, _ = build_prompt(args.request, trigger)
    print(f"Prompt: {prompt}")

    started = time.time()
    image, ext = generate_image(prompt, age=age, lora_scale=args.lora_scale, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"age-{age}-{int(started)}.{ext}")
    with open(path, "wb") as f:
        f.write(image)
    print(f"Saved {path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
