# ✨ Assignment 3 — Transformers, Self-Supervision & Generative Models

> 🚦 **Status:** ⬜⬜⬜⬜ Not started yet — the grand finale. 🎆

Modern deep learning in one assignment: **attention**, **self-supervised learning**, **diffusion**, and
**vision-language models**. This is where the magic I've been reading about actually happens — and where
my laptop starts sweating. 🥵 GPU recommended, patience mandatory.

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | `TransformerCaptioning.ipynb` | Transformer captioner: multi-head attention, positional encodings, train on COCO 🤖 | ⬜ |
| 2 | `SelfSupervisedLearning.ipynb` | SimCLR: contrastive pretraining → linear probe, with a provided backbone 🧊 | ⬜ |
| 3 | `DDPM.ipynb` | Text-conditioned diffusion: forward/reverse processes, U-Net, train or load pretrained ✨ | ⬜ |
| 4 | `CLIPDINO.ipynb` | CLIP zero-shot classification + DINO features, video object tracking on DAVIS 🎥 | ⬜ |

## 🔑 Ideas I'm chasing

- Self-attention: Q/K/V and why multi-head beats single-head 🎯
- Positional encodings — how a permutation-sensitive model learns order
- Contrastive learning: pulling positives, pushing negatives 🧲
- Diffusion: gradually adding noise, then learning to reverse it 🌀

## 💭 Notes & takeaways

*(To be written as I go — e.g. why SimCLR wants a big batch, what the temperature does in the contrastive loss.)*

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| Transformer captioning | loss dropping + readable captions | 🔜 |
| SimCLR linear probe | solid accuracy with the pretrained backbone | 🔜 |
| DDPM | recognizable generated emoji-images 🍀 | 🔜 |
| CLIP zero-shot | sensible top-1 classes on the probe set | 🔜 |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment3 && jupyter notebook`).
3. Run the **first cell** — COCO / imagenet_val / emoji datasets auto-download as needed. ⬇️
4. Pretrained weights (SimCLR, DDPM) fetch themselves on first use. 🤖
5. `CLIPDINO.ipynb` needs `tensorflow` + `tensorflow-datasets` (DAVIS video) — already in the env. 🎥

> ⚠️ Training parts really want a GPU. On a Mac, `mps` helps; for DDPM, just `git clone` this repo onto
> a rented GPU box (e.g. AutoDL) — everything is local-ready. 🚀

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| `cs231n/transformer_layers.py` | multi-head attention + positional encoding 🤖 |
| `cs231n/captioning_solver_transformer.py` | transformer training loop |
| `cs231n/simclr/` | contrastive-learning utils 🧊 |
| `cs231n/emoji_dataset.py` | DDPM's emoji dataset (auto-downloads) 🍀 |
| `cs231n/gaussian_diffusion.py` | the diffusion math 🌀 |
| `cs231n/unet.py` | the denoising U-Net |
| `cs231n/ddpm_trainer.py` | DDPM training + pretrained loader |
| `cs231n/clip_dino.py` | CLIP/DINO helpers (+ TFDS for DAVIS) 🎥 |
| `cs231n/coco_utils.py` | COCO loader |
| `data/`, `pretrained_model/` | where datasets & weights land ⬇️ |
| `collect_submission.ipynb` | 📦 zip + PDF for submission |

> ⚠️ Ignore `requirements.txt` — it's Colab-era and outdated; use the repo-level `env.yml`.

---

Finish line in sight! 🏁

