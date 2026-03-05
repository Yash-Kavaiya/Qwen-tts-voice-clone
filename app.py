#!/usr/bin/env python3
"""
app.py — Gradio Web UI for Qwen3-TTS Voice Cloning
===================================================
Launch with:
  python app.py [--port 7860] [--device cuda:0] [--share]

Features:
  Tab 1 — Voice Clone:   Upload your voice + transcript → speak new text
  Tab 2 — Voice Design:  Describe a voice in plain English → generate speech
  Tab 3 — Design+Clone:  Design a persona, then generate consistent lines
"""

import argparse
import tempfile
from pathlib import Path
from functools import lru_cache

import torch
import numpy as np
import soundfile as sf
import gradio as gr


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"  # no 0.6B variant available

SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Cloning Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect cuda:0 or cpu)")
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


ARGS = parse_args()

# Auto-detect device and dtype
if ARGS.device is None:
    ARGS.device = "cuda:0" if torch.cuda.is_available() else "cpu"
if ARGS.dtype is None:
    ARGS.dtype = "bfloat16" if "cuda" in ARGS.device else "float32"

print(f"🖥️  Device: {ARGS.device}  |  Dtype: {ARGS.dtype}")
if ARGS.device == "cpu":
    print("⚠  Running on CPU — inference will be slow but functional.")

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _get_attn_impl() -> str:
    if "cuda" in ARGS.device:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "eager"


# ═══════════════════════════════════════════════════════════════
# Model Loading (lazy, cached)
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=2)
def load_model(model_id: str):
    from qwen_tts import Qwen3TTSModel
    print(f"🔄 Loading {model_id}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=ARGS.device,
        dtype=DTYPE_MAP[ARGS.dtype],
        attn_implementation=_get_attn_impl(),
    )
    print(f"✅ {model_id} loaded.")
    return model


def _wav_to_gradio(wav: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Convert a float wav to int16 for Gradio audio output."""
    wav_int16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
    return (sr, wav_int16)


def _save_temp_wav(wav: np.ndarray, sr: int) -> str:
    """Save wav to a temp file for download."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, sr)
    return tmp.name


# ═══════════════════════════════════════════════════════════════
# Tab 1: Voice Clone
# ═══════════════════════════════════════════════════════════════

def voice_clone(
    ref_audio_path: str,
    ref_text: str,
    target_text: str,
    language: str,
    x_vector_only: bool,
):
    if not ref_audio_path:
        raise gr.Error("Please upload a reference audio file.")
    if not target_text.strip():
        raise gr.Error("Please enter the text to synthesize.")
    if not x_vector_only and not ref_text.strip():
        raise gr.Error("Please enter the reference transcript (or enable x-vector only mode).")

    model = load_model(CLONE_MODEL_ID)

    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text if not x_vector_only else None,
        x_vector_only_mode=x_vector_only,
    )

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language=language,
        voice_clone_prompt=prompt,
    )

    audio_out = _wav_to_gradio(wavs[0], sr)
    download_path = _save_temp_wav(wavs[0], sr)
    return audio_out, download_path


# ═══════════════════════════════════════════════════════════════
# Tab 2: Voice Design
# ═══════════════════════════════════════════════════════════════

def voice_design(
    target_text: str,
    instruct: str,
    language: str,
):
    if not target_text.strip():
        raise gr.Error("Please enter the text to speak.")
    if not instruct.strip():
        raise gr.Error("Please describe the desired voice.")

    model = load_model(DESIGN_MODEL_ID)

    wavs, sr = model.generate_voice_design(
        text=target_text,
        language=language,
        instruct=instruct,
    )

    audio_out = _wav_to_gradio(wavs[0], sr)
    download_path = _save_temp_wav(wavs[0], sr)
    return audio_out, download_path


# ═══════════════════════════════════════════════════════════════
# Tab 3: Design → Clone
# ═══════════════════════════════════════════════════════════════

def design_then_clone(
    persona: str,
    ref_sentence: str,
    ref_language: str,
    target_text: str,
    target_language: str,
):
    if not persona.strip():
        raise gr.Error("Please describe the voice persona.")
    if not ref_sentence.strip():
        raise gr.Error("Please enter a reference sentence for the designed voice.")
    if not target_text.strip():
        raise gr.Error("Please enter target text to generate.")

    # Step 1: Design the voice
    design_model = load_model(DESIGN_MODEL_ID)
    ref_wavs, sr = design_model.generate_voice_design(
        text=ref_sentence,
        language=ref_language,
        instruct=persona,
    )

    # Step 2: Clone the designed voice
    clone_model = load_model(CLONE_MODEL_ID)
    voice_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr),
        ref_text=ref_sentence,
    )

    wavs, sr = clone_model.generate_voice_clone(
        text=target_text,
        language=target_language,
        voice_clone_prompt=voice_prompt,
    )

    ref_audio_out = _wav_to_gradio(ref_wavs[0], sr)
    clone_audio_out = _wav_to_gradio(wavs[0], sr)
    download_path = _save_temp_wav(wavs[0], sr)
    return ref_audio_out, clone_audio_out, download_path


# ═══════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin-bottom: 0;
}
.sub-title {
    text-align: center;
    color: #6b7280;
    font-size: 1.05em;
    margin-top: 4px;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Qwen3-TTS Voice Studio", theme=gr.themes.Soft()) as app:

        gr.HTML('<h1 class="main-title">🎙️ Qwen3-TTS Voice Studio</h1>')
        gr.HTML('<p class="sub-title">Clone your voice · Design new voices · Generate expressive speech</p>')

        # ── Tab 1: Voice Clone ───────────────────────────────
        with gr.Tab("🎤 Voice Clone", id="clone"):
            gr.Markdown(
                "### Clone Your Voice\n"
                "Upload a short recording of your voice (5-30 seconds, WAV recommended) "
                "along with its transcript, then type new text to hear it in your voice."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    clone_ref_audio = gr.Audio(
                        label="📁 Reference Audio (your voice)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    clone_ref_text = gr.Textbox(
                        label="📝 Reference Transcript",
                        placeholder="Type what was said in the reference audio...",
                        lines=3,
                    )
                    clone_xvec = gr.Checkbox(
                        label="⚡ x-vector only (no transcript needed, lower quality)",
                        value=False,
                    )
                with gr.Column(scale=1):
                    clone_target = gr.Textbox(
                        label="🗣️ Text to Synthesize",
                        placeholder="Type the text you want to say in your cloned voice...",
                        lines=5,
                    )
                    clone_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="Auto", label="🌐 Language"
                    )
                    clone_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            clone_audio_out = gr.Audio(label="🔊 Generated Speech", interactive=False)
            clone_download = gr.File(label="📥 Download WAV")

            clone_btn.click(
                fn=voice_clone,
                inputs=[clone_ref_audio, clone_ref_text, clone_target, clone_lang, clone_xvec],
                outputs=[clone_audio_out, clone_download],
            )

        # ── Tab 2: Voice Design ──────────────────────────────
        with gr.Tab("🎨 Voice Design", id="design"):
            gr.Markdown(
                "### Design a Voice from Description\n"
                "Describe the voice you want in natural language — age, gender, tone, "
                "emotion, accent — and the model will synthesize speech matching that description."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    design_instruct = gr.Textbox(
                        label="🎭 Voice Description",
                        placeholder=(
                            "e.g. Young woman, 25 years old, warm alto voice, "
                            "slightly breathy, friendly and energetic"
                        ),
                        lines=4,
                    )
                with gr.Column(scale=1):
                    design_text = gr.Textbox(
                        label="🗣️ Text to Speak",
                        placeholder="Type the text you want this voice to say...",
                        lines=4,
                    )
                    design_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="Auto", label="🌐 Language"
                    )
            design_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            design_audio_out = gr.Audio(label="🔊 Designed Voice Output", interactive=False)
            design_download = gr.File(label="📥 Download WAV")

            design_btn.click(
                fn=voice_design,
                inputs=[design_text, design_instruct, design_lang],
                outputs=[design_audio_out, design_download],
            )

        # ── Tab 3: Design → Clone ────────────────────────────
        with gr.Tab("🔗 Design → Clone", id="design_clone"):
            gr.Markdown(
                "### Design a Persona then Clone It\n"
                "1. **Design**: Describe a persona and provide a reference sentence.\n"
                "2. **Clone**: The model creates a reference clip, then uses it to "
                "generate your target text in the same consistent voice. "
                "Perfect for character voice acting!"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    dc_persona = gr.Textbox(
                        label="🎭 Persona Description",
                        placeholder="Male, 30s, warm baritone, calm and reassuring therapist voice",
                        lines=3,
                    )
                    dc_ref_sentence = gr.Textbox(
                        label="📝 Reference Sentence",
                        placeholder="A short sentence the designed voice will say first...",
                        lines=2,
                    )
                    dc_ref_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="English", label="🌐 Ref Language"
                    )
                with gr.Column(scale=1):
                    dc_target = gr.Textbox(
                        label="🗣️ Target Text to Generate",
                        placeholder="Type the text you want in the designed+cloned voice...",
                        lines=5,
                    )
                    dc_target_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="English", label="🌐 Target Language"
                    )
            dc_btn = gr.Button("🚀 Design & Clone", variant="primary", size="lg")

            with gr.Row():
                dc_ref_audio = gr.Audio(label="🎨 Designed Reference Clip", interactive=False)
                dc_clone_audio = gr.Audio(label="🔊 Cloned Output", interactive=False)
            dc_download = gr.File(label="📥 Download Cloned WAV")

            dc_btn.click(
                fn=design_then_clone,
                inputs=[dc_persona, dc_ref_sentence, dc_ref_lang, dc_target, dc_target_lang],
                outputs=[dc_ref_audio, dc_clone_audio, dc_download],
            )

        # ── Footer ───────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "**Models**: Qwen3-TTS-12Hz-1.7B-Base (clone) · Qwen3-TTS-12Hz-1.7B-VoiceDesign (design)  \n"
            "**Languages**: Chinese, English, Japanese, Korean, German, French, "
            "Russian, Portuguese, Spanish, Italian  \n"
            "Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) + Gradio"
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_port=ARGS.port,
        share=ARGS.share,
        server_name="0.0.0.0",
    )
