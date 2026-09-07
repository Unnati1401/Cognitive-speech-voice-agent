#!/usr/bin/env python
"""Generate demo sample clips for the web app (assets/samples/).

Uses macOS `say` to synthesize a few picture-description clips long enough to
clear the 30s minimum, so website testers can click a sample instead of
uploading their own voice.

    uv run python scripts/make_samples.py     # writes assets/samples/*.wav
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "samples"

# label -> long-ish description (fluent vs. hesitant), each ~30-45s when spoken
SAMPLES = {
    "sample_fluent": (
        "In this kitchen scene a little boy is standing on a tall stool trying to "
        "reach the cookie jar on the high shelf, and the stool is wobbling and about "
        "to tip over. His younger sister stands beside him with her hand held up, "
        "waiting for a cookie to be passed down to her. Behind them their mother is "
        "standing at the sink washing and drying the dishes, but she is looking away "
        "and does not notice that the water is overflowing from the sink and spilling "
        "onto the kitchen floor. The window is open and the curtains are gently blowing "
        "in the breeze, and outside you can see a path and a garden."
    ),
    "sample_hesitant": (
        "Um there is a there is a boy and uh he is on the on the thing, the stool I think, "
        "and um he is reaching for the uh the cookies, the jar. And there is a a girl, "
        "his sister maybe, and she is uh she wants one too. And um the the mother is "
        "there, she is doing something with the with the water, and uh I think the water "
        "is is coming out, it is uh spilling, and she does not she does not see it. And "
        "um there is a window and uh some other things, I am not sure, some stuff over there."
    ),
    "sample_moderate": (
        "There is a boy climbing up on a stool to get some cookies from a jar up high. "
        "The stool looks like it is tipping. A girl is next to him reaching up as well. "
        "The mother is by the sink and the water is running over the side onto the floor, "
        "but she is drying a plate and looking out the window and does not notice. Um it "
        "is a busy kitchen and a few things are going wrong at the same time."
    ),
}


def main() -> int:
    if sys.platform != "darwin":
        print("This helper uses macOS `say`. On other systems, drop your own .wav "
              f"files into {OUT} instead.", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES.items():
        out = OUT / f"{name}.wav"
        subprocess.run(
            ["say", "-o", str(out), "--data-format=LEI16@16000", text], check=True)
        print(f"wrote {out}", file=sys.stderr)
    print(f"\nDone. {len(SAMPLES)} samples in {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())