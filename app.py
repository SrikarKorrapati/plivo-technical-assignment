"""Down Memory Lane Slack bot: @mention it or DM it, get a childhood photo back in the thread."""

import logging
import os
import re
import threading
from collections import OrderedDict

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from imagegen import build_prompt, generate_image

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("down-memory-lane")

app = App(token=os.environ["SLACK_BOT_TOKEN"])
TRIGGER_WORD = os.getenv("TRIGGER_WORD", "TOK")
MENTION = re.compile(r"<@[A-Z0-9]+>")

HELP_TEXT = (
    "Tell me which memory to create, e.g. _\"my 5-year-old self on a beach\"_ "
    "or _\"my 2-year-old self in my house's backyard\"_."
)

# Slack can redeliver an event; remember recently handled messages so we generate once.
_seen = OrderedDict()
_seen_lock = threading.Lock()


def first_time(key):
    with _seen_lock:
        if key in _seen:
            return False
        _seen[key] = True
        while len(_seen) > 1000:
            _seen.popitem(last=False)
        return True


def handle_request(event, client):
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    text = MENTION.sub("", event.get("text", "")).strip()
    # One message can hold several requests, one per line; each gets its own image.
    requests = [line.strip() for line in text.splitlines() if line.strip(" \t\"“”'.,")]

    if not requests:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=HELP_TEXT)
        return

    try:
        client.reactions_add(channel=channel, timestamp=event["ts"], name="hourglass_flowing_sand")
    except Exception:
        log.warning("Could not add reaction", exc_info=True)

    # Generated one at a time: low-credit Replicate accounts allow one prediction at once.
    results = [create_memory(client, channel, thread_ts, request) for request in requests]
    reaction = "white_check_mark" if all(results) else "x"

    try:
        client.reactions_remove(channel=channel, timestamp=event["ts"], name="hourglass_flowing_sand")
        client.reactions_add(channel=channel, timestamp=event["ts"], name=reaction)
    except Exception:
        log.warning("Could not update reactions", exc_info=True)


def create_memory(client, channel, thread_ts, text):
    """Generate one image for one request and post it in the thread. Returns True on success."""
    prompt, age, setting = build_prompt(text, TRIGGER_WORD)
    log.info("Request in %s: %r -> age=%s setting=%r", channel, text, age, setting)

    status = client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=f":art: Travelling back to when you were {age}... ({setting}). This takes about 30 seconds.",
    )

    try:
        image, ext = generate_image(prompt, age=age)
        client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            content=image,
            filename=f"memory-age-{age}.{ext}",
            title=f"You at {age}: {setting}",
            alt_txt=f"AI-generated photo of the requester as a {age}-year-old, {setting}",
            initial_comment=f"Here you are at {age}, {setting} :sparkles:",
        )
        client.chat_delete(channel=channel, ts=status["ts"])
        return True
    except Exception:
        log.exception("Generation failed for %r", text)
        client.chat_update(
            channel=channel,
            ts=status["ts"],
            text=":warning: Sorry, I couldn't create that memory. Please try again in a minute, "
                 "or rephrase the request.",
        )
        return False


@app.event("app_mention")
def on_mention(event, client):
    if event.get("bot_id") or not first_time(event.get("client_msg_id") or event["ts"]):
        return
    handle_request(event, client)


@app.event("message")
def on_message(event, client):
    # Only plain user DMs; channel messages arrive as app_mention, edits/bot posts have a subtype.
    if event.get("channel_type") != "im" or event.get("subtype") or event.get("bot_id"):
        return
    if not first_time(event.get("client_msg_id") or event["ts"]):
        return
    handle_request(event, client)


if __name__ == "__main__":
    log.info("Starting Down Memory Lane bot (Socket Mode)")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
