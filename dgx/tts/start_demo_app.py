#!/usr/bin/env python3

import argparse
import subprocess
import sys

TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

MODELS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

HOST = "0.0.0.0"
PORT = "8000"


def run(cmd):
    print(">>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=str, default=PORT)

    parser.add_argument("--no-download", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base", action="store_true")
    group.add_argument("--voice-design", action="store_true")
    group.add_argument("--custom-voice", action="store_true")

    args = parser.parse_args()

    if args.base:
        mode = "base"
    elif args.custom_voice:
        mode = "custom-voice"
    else:
        mode = "voice-design"  # default

    model = MODELS[mode]

    print(f"Selected mode: {mode}")
    print(f"Model: {model}")

    if not args.no_download:
        print("Downloading tokenizer...")
        run(["hf", "download", TOKENIZER])

        print("Downloading model...")
        run(["hf", "download", model])

    print("Starting server...")
    os_exec = ["qwen-tts-demo", model, "--ip", args.host, "--port", args.port]

    # replace process (proper signal handling)
    import os
    os.execvp(os_exec[0], os_exec)


if __name__ == "__main__":
    main()
