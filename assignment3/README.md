# ✨ Assignment 3 — Transformers, Self-Supervision & Generative Models

> 🚦 **Status:** ✅🟨⬜⬜ **1 / 4 done + exercise 2 code-complete — GPU training still to go.** ⏳

Modern deep learning in one assignment: **attention**, **self-supervised learning**, **diffusion**, and
**vision-language models**. This is where the magic I've been reading about actually happens — and where
my laptop starts sweating. 🥵 GPU recommended, patience mandatory.

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | [`TransformerCaptioning.ipynb`](TransformerCaptioning.ipynb) | Transformer captioner: multi-head attention, positional encodings, train on COCO 🤖 + a small Vision Transformer on CIFAR-10 👁️ | ✅ |
| 2 | [`SelfSupervisedLearning.ipynb`](SelfSupervisedLearning.ipynb) | SimCLR: contrastive pretraining → linear probe, with a provided backbone 🧊 | 🟨 code ✅ · GPU training 🔜 |
| 3 | [`DDPM.ipynb`](DDPM.ipynb) | Text-conditioned diffusion: forward/reverse processes, U-Net, train or load pretrained ✨ | ⬜ |
| 4 | [`CLIPDINO.ipynb`](CLIPDINO.ipynb) | CLIP zero-shot classification + DINO features, video object tracking on DAVIS 🎥 | ⬜ |

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
| Transformer captioning | loss dropping + readable captions — overfit 50 imgs to **0.0225**, every layer test within tolerance | ✅ |
| Vision Transformer on CIFAR-10 | > 0.45 test acc in 2 epochs — got **0.5124** (patch 4, lr 5e-4, wd 1e-4, bs 16, 6 layers); one-batch overfit **1.00** | ✅ |
| SimCLR code (no training) | every sanity check passes — augmentation error **0**, sim / naive+vectorized loss errors ≤ **5.7e-8**, `train()` smoke-tested on 4 imgs (loss 1.667, weights really updated) | ✅ |
| SimCLR + linear probe | ≥ 70% top-1 with the pretrained backbone vs a from-scratch baseline — **not run: GPU training deferred** ⏳ | 🔜 |
| DDPM | recognizable generated emoji-images 🍀 | 🔜 |
| CLIP zero-shot | sensible top-1 classes on the probe set | 🔜 |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment3 && jupyter notebook`).
3. Run the **first cell** — COCO / imagenet_val / emoji datasets auto-download as needed. ⬇️
4. Pretrained weights (SimCLR, DDPM) fetch themselves on first use. 🤖
5. [`CLIPDINO.ipynb`](CLIPDINO.ipynb) needs `tensorflow` + `tensorflow-datasets` (DAVIS video) — already in the env. 🎥
6. In [`TransformerCaptioning.ipynb`](TransformerCaptioning.ipynb) the only heavy cell is the last one — a 2-epoch ViT on the full CIFAR-10, a few minutes on CPU, and it uses `cuda` automatically when there is one. ⏳
7. [`SelfSupervisedLearning.ipynb`](SelfSupervisedLearning.ipynb)'s training cells still hard-code `device='cuda'`; before running them locally, swap those for the notebook's `device` variable (otherwise they only run on a CUDA box). 🔧

> 💻 Only the transformer notebook has been run **locally, no GPU rental** — the ViT still cleared the bar at **0.5124** test accuracy. ⏳ The SimCLR code is done and passes every sanity check, but its 1-epoch pretrain + linear-probe runs **have not happened yet**; DDPM and CLIP/DINO aren't started either. Nothing below exercise 1 has any real GPU training behind it — next stop: a rented GPU box (e.g. AutoDL), the repo is local-ready. 🚀

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| [`cs231n/transformer_layers.py`](cs231n/transformer_layers.py) | multi-head attention, positional encoding, decoder/encoder layers, patch embedding 🤖 |
| [`cs231n/classifiers/transformer.py`](cs231n/classifiers/transformer.py) | captioning Transformer + Vision Transformer 📸 |
| [`cs231n/captioning_solver_transformer.py`](cs231n/captioning_solver_transformer.py) | transformer training loop |
| [`cs231n/simclr/`](cs231n/simclr) | contrastive-learning utils 🧊 |
| [`cs231n/emoji_dataset.py`](cs231n/emoji_dataset.py) | DDPM's emoji dataset (auto-downloads) 🍀 |
| [`cs231n/gaussian_diffusion.py`](cs231n/gaussian_diffusion.py) | the diffusion math 🌀 |
| [`cs231n/unet.py`](cs231n/unet.py) | the denoising U-Net |
| [`cs231n/ddpm_trainer.py`](cs231n/ddpm_trainer.py) | DDPM training + pretrained loader |
| [`cs231n/clip_dino.py`](cs231n/clip_dino.py) | CLIP/DINO helpers (+ TFDS for DAVIS) 🎥 |
| [`cs231n/coco_utils.py`](cs231n/coco_utils.py) | COCO loader |
| `data/`, `pretrained_model/` | where datasets & weights land (local, git-ignored) ⬇️ |
| [`collect_submission.ipynb`](collect_submission.ipynb) | 📦 zip + PDF for submission |

> ⚠️ Ignore [`requirements.txt`](requirements.txt) — it's Colab-era and outdated; use the repo-level [`env.yml`](../env.yml).

---

Finish line in sight! 🏁

