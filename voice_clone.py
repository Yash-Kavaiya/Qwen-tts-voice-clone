#!/usr/bin/env python3
"""
voice_clone.py — Clone your voice with Qwen3-TTS
=================================================
Provide a short reference audio of your voice (+ its transcript)
and this script will synthesize NEW text in your cloned voice.

Usage:
  # Basic — clone from a local WAV file
  python voice_clone.py \
      --ref-audio my_voice.wav \
      --ref-text "Hello, this is me speaking naturally." \
      --text "Qwen three TTS can clone my voice perfectly!"

  # Clone from a URL
  python voice_clone.py \
      --ref-audio https://example.com/sample.wav \
      --ref-text "Okay. Yeah. I resent you." \
      --text "This is a brand new sentence in the cloned voice."

  # x-vector only mode (no transcript needed, lower quality)
  python voice_clone.py \
      --ref-audio my_voice.wav \
      --x-vector-only \
      --text "Quick clone without a transcript."

  # Multi-sentence batch
  python voice_clone.py \
      --ref-audio my_voice.wav \
      --ref-text "This is my reference." \
      --text "Sentence one." "Sentence two." "Sentence three." \
      --language English English English

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
Hardware: NVIDIA GPU recommended; CPU supported (slower)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone your voice using Qwen3-TTS Base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Path or URL to a reference audio clip of your voice (WAV recommended).",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of the reference audio. Required unless --x-vector-only is set.",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="One or more sentences to synthesize in the cloned voice.",
    )
    parser.add_argument(
        "--language",
        nargs="+",
        default=None,
        help=(
            "Language(s) for each sentence. Supported: Chinese, English, Japanese, "
            "Korean, German, French, Russian, Portuguese, Spanish, Italian. "
            "Defaults to 'Auto' (auto-detect)."
        ),
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use only the speaker embedding (no transcript needed). Faster but lower quality.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model name or local path (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: auto-detect cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16 on GPU, float32 on CPU).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save generated WAV files (default: output/).",
    )
    parser.add_argument(
        "--output-prefix",
        default="clone",
        help="Filename prefix for output files (default: clone).",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()

    # ── Auto-detect device and dtype ─────────────────────────
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "bfloat16" if "cuda" in args.device else "float32"

    # ── Validate inputs ──────────────────────────────────────
    if not args.x_vector_only and args.ref_text is None:
        print("❌ --ref-text is required unless --x-vector-only is set.")
        sys.exit(1)

    texts = args.text
    if args.language:
        if len(args.language) == 1 and len(texts) > 1:
            languages = args.language * len(texts)
        elif len(args.language) != len(texts):
            print(f"❌ Number of --language values ({len(args.language)}) must match --text ({len(texts)}).")
            sys.exit(1)
        else:
            languages = args.language
    else:
        languages = ["Auto"] * len(texts)

    # ── Output directory ─────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────
    print(f"🔄 Loading model: {args.model}")
    print(f"   Device: {args.device}  |  Dtype: {args.dtype}")
    if args.device == "cpu":
        print("⚠  Running on CPU — this will be slow but functional.")

    from qwen_tts import Qwen3TTSModel

    attn_impl = "eager"
    if "cuda" in args.device:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            print("⚠  Flash Attention 2 not found — using SDPA.")
    else:
        print("   Using eager attention (CPU mode).")

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=get_dtype(args.dtype),
        attn_implementation=attn_impl,
    )
    print("✅ Model loaded.\n")

    # ── Build reusable voice-clone prompt ─────────────────────
    print("🎙️  Building voice-clone prompt from reference audio...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text if not args.x_vector_only else None,
        x_vector_only_mode=args.x_vector_only,
    )
    print("✅ Voice-clone prompt ready.\n")

    # ── Generate speech ──────────────────────────────────────
    if len(texts) == 1:
        print(f'🗣️  Generating: "{texts[0]}"')
        wavs, sr = model.generate_voice_clone(
            text=texts[0],
            language=languages[0],
            voice_clone_prompt=prompt,
        )
    else:
        print(f"🗣️  Generating batch of {len(texts)} sentences...")
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=prompt,
        )

    # ── Save outputs ─────────────────────────────────────────
    for i, wav in enumerate(wavs):
        filename = f"{args.output_prefix}_{i + 1}.wav"
        filepath = Path(args.output_dir) / filename
        sf.write(str(filepath), wav, sr)
        print(f"   💾 Saved: {filepath}")

    print(f"\n✅ Done! {len(wavs)} file(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
