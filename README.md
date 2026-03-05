# 🎙️ Qwen3-TTS Voice Clone

Clone your voice using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — an open-source TTS model by Alibaba Cloud's Qwen team supporting **voice cloning**, **voice design**, and **expressive speech generation** in 10+ languages.

## ✨ Features

| Feature | Script | Model |
|---|---|---|
| **Voice Clone** — clone from a reference audio | `voice_clone.py` | `Qwen3-TTS-12Hz-1.7B-Base` |
| **Voice Design** — create a voice from a text description | `voice_design.py` | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| **Design → Clone** — design a persona, then reuse it | `design_then_clone.py` | Both models |
| **Web UI** — all features in a Gradio browser interface | `app.py` | Both models |

### 🌐 Supported Languages
Chinese · English · Japanese · Korean · German · French · Russian · Portuguese · Spanish · Italian

## 🖥️ Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 24 GB (e.g. RTX 4090) |
| System RAM | 16 GB | 32 GB |
| CUDA | 11.8+ | 12.x |
| Python | 3.10 | 3.12 |

> **Note**: Flash Attention 2 is optional but recommended for lower VRAM usage and faster inference.

## 🚀 Quick Setup

### Option A: Automated (Linux/macOS)
```bash
chmod +x setup.sh
./setup.sh
```

### Option B: Manual
```bash
# Create environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install dependencies
pip install -U qwen-tts soundfile numpy gradio

# (Optional) Flash Attention 2
pip install -U flash-attn --no-build-isolation
```

### Option C: pip only
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Voice Clone (clone your voice from a recording)

```bash
# Record a 5-30 second clip of your voice → my_voice.wav

python voice_clone.py \
    --ref-audio my_voice.wav \
    --ref-text "Hello, this is me speaking naturally for the reference." \
    --text "Now this brand new sentence will sound like me!" \
    --language English
```

**Options:**
- `--x-vector-only` — skip transcript (faster, lower quality)
- `--text "A" "B" "C"` — batch generate multiple sentences
- `--language English Chinese` — per-sentence language
- `--output-dir output/` — output directory

### 2. Voice Design (create a voice from a description)

```bash
python voice_design.py \
    --text "Welcome to our AI podcast!" \
    --instruct "Young woman, 25, warm and friendly, slightly breathy alto" \
    --language English
```

### 3. Design → Clone (design a persona, reuse it)

```bash
python design_then_clone.py \
    --persona "Male, 30s, calm baritone, reassuring therapist voice" \
    --ref-sentence "Hello, welcome. How are you feeling today?" \
    --text "Let's start with some breathing exercises." \
         "Take a deep breath in... and slowly out." \
    --language English
```

### 4. Web UI (browser interface)

```bash
python app.py --port 7860

# Public link (for remote access):
python app.py --share
```

Then open **http://localhost:7860** in your browser.

| Tab | What it does |
|---|---|
| 🎤 Voice Clone | Upload audio + transcript → generate new speech |
| 🎨 Voice Design | Describe a voice → generate speech |
| 🔗 Design → Clone | Design a persona → clone for consistent lines |

## 📁 Project Structure

```
Qwen-tts-voice-clone/
├── app.py                  # Gradio web UI
├── voice_clone.py          # CLI voice cloning
├── voice_design.py         # CLI voice design
├── design_then_clone.py    # CLI design + clone pipeline
├── setup.sh                # Environment setup script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── output/                 # Generated audio files (auto-created)
```

## 🔗 Links

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS-12Hz-1.7B-Base (HuggingFace)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign (HuggingFace)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Technical Report](https://arxiv.org/abs/2601.15621)

## 📄 Citation

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and others},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## 📝 License

This project uses the Qwen3-TTS models which are released under the [Apache 2.0 License](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE).