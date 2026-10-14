"""Prompt building and Replicate image generation, shared by app.py and generate.py."""

import os
import re
import time

import replicate
from replicate.exceptions import ReplicateError

DEFAULT_AGE = 5

AGE_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# "2-year-old", "2 year old", "two years old", "age 7", "aged 10"
AGE_PATTERN = re.compile(
    r"\b(\d{1,2}|" + "|".join(AGE_WORDS) + r")[\s-]*(?:years?|yrs?)[\s-]*old\b"
    r"|\bage[d]?\s+(\d{1,2}|" + "|".join(AGE_WORDS) + r")\b",
    re.IGNORECASE,
)


def parse_age(text):
    match = AGE_PATTERN.search(text)
    if not match:
        return DEFAULT_AGE
    raw = (match.group(1) or match.group(2)).lower()
    age = AGE_WORDS.get(raw) or int(raw)
    return max(1, min(age, 17))


def describe_age(age):
    # SUBJECT_GENDER ("boy"/"girl") keeps the child's gender from drifting at low LoRA strength.
    gender = os.getenv("SUBJECT_GENDER", "").strip()
    kid = f"{gender}" if gender else "child"
    if age <= 3:
        return f"a {age}-year-old toddler {kid}, chubby cheeks, small toddler body proportions"
    if age <= 7:
        return f"a {age}-year-old young {kid}, small child body proportions, round face"
    if age <= 12:
        return f"a {age}-year-old {kid}"
    return f"a {age}-year-old teenage {kid}" if gender else f"a {age}-year-old teenager"


def extract_setting(text, age):
    """Strip the age phrase and filler like 'my/your ... self' to leave the scene."""
    scene = re.sub(r"[\"“”]", " ", AGE_PATTERN.sub(" ", text)).strip()
    scene = re.sub(r"\b(my|your)\b", "the", scene, flags=re.IGNORECASE)
    scene = re.sub(r"\b(self|me|myself|yourself|photo of|picture of|image of|generate|create|show)\b",
                   " ", scene, flags=re.IGNORECASE)
    scene = re.sub(r"^\s*the\s+", " ", scene, flags=re.IGNORECASE)  # leftover from "my 5-year-old self"
    scene = re.sub(r"\bat\s+(?=\w+ing\b)", " ", scene, flags=re.IGNORECASE)  # "at age 7 riding" -> "riding"
    scene = re.sub(r"\s+", " ", scene).strip(" ,.!?-")
    return scene or "at home"


def build_prompt(text, trigger_word):
    age = parse_age(text)
    setting = extract_setting(text, age)
    # Age-independent traits (skin tone, eyes, hair) keep the likeness when the LoRA is
    # weakened for young ages; adult-only traits like glasses or a beard must not go here.
    traits = os.getenv("SUBJECT_DESCRIPTION", "").strip()
    traits = f" with {traits}" if traits else ""
    prompt = (
        f"A candid childhood photo of {trigger_word} as {describe_age(age)}{traits}, {setting}. "
        f"Same face, eyes, skin tone and features as {trigger_word}, but clearly a young child aged {age}. "
        "Nostalgic family snapshot, natural light, realistic skin, 35mm film photo, "
        "soft film grain, slightly faded colors, sharp focus on the child's face."
    )
    return prompt, age, setting


YOUNG_CHILD_LORA_SCALE = 0.65


def default_lora_scale(age):
    """A full-strength LoRA pulls young faces toward the adult training photos; ease off for small kids."""
    if isinstance(age, int) and age <= 7:
        return YOUNG_CHILD_LORA_SCALE
    return float(os.getenv("LORA_SCALE", "0.8"))


def generate_image(prompt, age=None, lora_scale=None, seed=None):
    """Run the trained Flux LoRA on Replicate and return (image_bytes, extension)."""
    model = os.environ["REPLICATE_MODEL"]
    output_format = "jpg"
    inputs = {
        "prompt": prompt,
        "lora_scale": float(lora_scale if lora_scale is not None else default_lora_scale(age)),
        "num_outputs": 1,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "aspect_ratio": "1:1",
        "output_format": output_format,
        "output_quality": 90,
        # go_fast multiplies lora_scale by 1.5, which makes tuning unpredictable.
        "go_fast": False,
    }
    if seed is not None:
        inputs["seed"] = int(seed)
    # Accounts with under $5 credit may only start one prediction at a time (HTTP 429),
    # so back off and retry instead of failing when two requests arrive together.
    for attempt in range(6):
        try:
            output = replicate.run(model, input=inputs)
            break
        except ReplicateError as err:
            if getattr(err, "status", None) != 429 or attempt == 5:
                raise
            time.sleep(10)
    first = output[0] if isinstance(output, (list, tuple)) else output
    # replicate>=1.0 returns FileOutput objects; read the bytes now because output URLs expire.
    return first.read(), output_format
