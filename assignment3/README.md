# ✨ Assignment 3 — Transformers, Self-Supervision & Generative Models

> 🚦 **Status:** ✅✅✅✅ **4 / 4 — Assignment 3 COMPLETE!** 🎉🏁

Modern deep learning in one assignment: **attention**, **self-supervised learning**, **diffusion**, and
**vision-language models**. This is where the magic I've been reading about actually happens. 🥵 And it
turned out to be a laptop-friendly assignment: attention, diffusion and CLIP/DINO all ran locally on CPU,
while SimCLR's contrastive pretraining was the one job worth renting a GPU for — an AutoDL
**RTX 4090 ×1**, where one epoch took 23 s. 💻

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | [`TransformerCaptioning.ipynb`](TransformerCaptioning.ipynb) | Transformer captioner: multi-head attention, positional encodings, train on COCO 🤖 + a small Vision Transformer on CIFAR-10 👁️ | ✅ |
| 2 | [`SelfSupervisedLearning.ipynb`](SelfSupervisedLearning.ipynb) | SimCLR: contrastive pretraining → linear probe, with a provided backbone 🧊 | ✅ |
| 3 | [`DDPM.ipynb`](DDPM.ipynb) | Text-conditioned diffusion: forward/reverse processes, U-Net, train or load pretrained ✨ | ✅ |
| 4 | [`CLIPDINO.ipynb`](CLIPDINO.ipynb) | CLIP zero-shot classification + DINO features, video object tracking on DAVIS 🎥 | ✅ |

## 🔑 Ideas I'm chasing

- Self-attention: Q/K/V and why multi-head beats single-head 🎯
- Positional encodings — how a permutation-sensitive model learns order
- Contrastive learning: pulling positives, pushing negatives 🧲
- Diffusion: gradually adding noise, then learning to reverse it 🌀

## 💭 Notes & takeaways

**Attention is a soft, learned lookup — and the mask polarity is a trap** 🤖
- Multi-head means several independent `d/h` subspaces, so heads can specialise (short-range vs long-range); the output projection is the only place the heads actually talk to each other 🎯
- `attn_mask` means "**True = keep**": masking the `True` entries instead gave a masked-attention error of **0.575**, while `masked_fill(~mask, -1e9)` gave **9.7e-5** 🩹 — hence the captioner's causal mask is `torch.tril(ones)`, not `tril == 0` 🔒
- `-1e9` rather than `-inf` on purpose: a fully masked row degrades to uniform instead of NaN

**Transformer captioner: causal masking is the whole trick** 🖼️➡️💬
- 50 images, 2 layers: loss **5.05 → 0.0225** in 100 epochs (bar < 0.05) while the decoder sees the whole caption at once — hiding every future step is what makes that a legal signal
- Vision Transformer on CIFAR-10: **0.5124** test accuracy in 2 epochs (bar 0.45) 📈
- In a 2-epoch budget patches beat capacity: 4×4 patches (64 tokens) > 8×8 (16 tokens) by ~2 points, while a wider hidden dim or FFN made it *worse* (0.35 / 0.32 vs 0.41) ⚖️

**Contrastive learning: the batch *is* the negative sampler** 🧲
- InfoNCE only compares in-batch neighbours, so N pairs give just 2(N−1) negatives per anchor — the reason CLIP used batch 32k and SimCLR 4k+, and why MoCo / BYOL / DINO / SigLIP exist at all
- The payoff, measured on a rented GPU: one extra pretraining epoch pushed the kNN eval to **83.54%** top-1, and a linear probe on the frozen features reached **82.40%** test top-1 versus **15.28%** for the same probe on a from-scratch backbone — **+67 points** with only 10% of the labels 🚀
- The code matched the reference keys at the float32 floor (augmentation error **0**, losses ≤ **5.7e-8**) — but `sim_positive_pairs` had to return **N×1**: the key file stores `answers['sim']` as `(2, 1)`, and `(N,)` silently broadcasts into a wrong comparison 🔬

**Diffusion: the network only ever learns to denoise one step** 🌀
- The forward process is closed-form, so training collapses to an MSE against whatever the net predicts — I implemented both directions (`pred_noise` ↔ `pred_x_start`) 🎯
- Structure is pinned by arithmetic: the U-Net's exact parameter count (**6,499** toy / **12.4 M** real) plus a strict `load_state_dict` of the pretrained checkpoint
- Sampling is cheap (**~3 s** for 100 steps × 5 images on CPU) and the text conditioning is real: generations have emoji-like statistics and CLIP ranks the prompt top-1 for all 5 samples 🎩

**CLIP and DINO: alignment is not localization** 🎥
- CLIP's zero-shot path is normalize → dot-product → argmax, and it works: retrieval returned tennis + skateboard for "sports" and bathroom + zebras for "black and white", with **9/10** zero-shot labels on target 🔎
- DINO's one-shot segmentation was the surprise: a 2-layer MLP trained on the patches of **one** annotated frame reached IoU **0.484 / 0.560 / 0.645** (bars 0.45 / 0.50 / 0.55) 🏆
- Why it works is measurable: neighbouring patch embeddings score **0.83** cosine similarity vs **0.47** for random pairs, so PCA-over-features paints object-shaped regions — a free segmentation 👁️
- CLIP optimises the *pooled* token for caption matching, DINO makes two crops of one object agree; for per-patch tasks that difference is everything 🧭

**Craft tips** 🔧
- Check the *absolute* difference before chasing a "failed" tolerance: three checks here were float32 rounding at the 1e-6 level ❌➡️✅
- Structural invariants (exact parameter counts, dimension assertions, strict checkpoint loads) are the real tests
- Local-first works, but the plumbing is what breaks: `pip install git+…` times out, `tfds` wanted a missing `importlib_resources`, `device='cuda'` was hard-coded — all now "install only if missing" or device-agnostic 💻

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| Transformer captioning | bar: loss < 0.05 when overfitting 50 images — reached **0.0225** in 100 epochs | ✅ |
| Vision Transformer on CIFAR-10 | bar: > 0.45 test acc after 2 epochs — got **0.5124** (patch 4, lr 5e-4, wd 1e-4, bs 16, 6 layers) | ✅ |
| DDPM emoji generation | text-conditioned emoji faces from the pretrained U-Net — **~3 s** per 100-step sample on CPU | ✅ |
| CLIP zero-shot | **9/10** sample images land on the expected class (the miss is a 5e-5 tie) | ✅ |
| CLIP retrieval | text → image search: "sports" → tennis + skateboard, "black and white" → bathroom + zebras | ✅ |
| DINO one-shot segmentation | bars: >0.45 / >0.50 / >0.55 mean IoU from a single annotated frame — got **0.484 / 0.560 / 0.645** | ✅ |
| SimCLR + linear probe | bar: ≥ 70% top-1 with the pretrained backbone — got **82.40%** vs **15.28%** for the from-scratch baseline, plus **83.54%** top-1 from the kNN eval after 1 pretraining epoch (AutoDL **RTX 4090 ×1**) | ✅ |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment3 && jupyter notebook`).
3. Run the **first cell** of each notebook — it locates `cs231n` and pulls in that notebook's data; the weights (SimCLR 99 MB → CLIP 338 MB) and DAVIS (794 MB) download themselves on first use, so the first run is slow and the rest are seconds. ⬇️
4. The only job worth a GPU was SimCLR's pretraining — everything else finished on CPU. It ran on an AutoDL **RTX 4090 ×1** (23 s per epoch). Now everything is done. 🏁

> 💻 Exercises 1, 3 and 4 ran **locally, no GPU rental** (ViT **0.5124** test accuracy, DDPM emojis in ~3 s per prompt, DINO one-shot segmentation **0.645** mean IoU), and exercise 2 ran on a rented AutoDL **RTX 4090 ×1**: a linear probe on SimCLR features jumped from **15.28%** to **82.40%** test top-1 with only 10% of the labels. 🚀

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
| [`cs231n/clip_dino.py`](cs231n/clip_dino.py) | CLIP similarity / zero-shot / retrieval + DINO attention, PCA and DAVIS one-shot segmentation 🎥 |
| [`cs231n/coco_utils.py`](cs231n/coco_utils.py) | COCO loader |
| `data/`, `pretrained_model/` | where datasets & weights land (local, git-ignored); DAVIS goes to `~/tensorflow_datasets` ⬇️ |
| [`collect_submission.ipynb`](collect_submission.ipynb) | 📦 zip + PDF for submission |

> ⚠️ Ignore [`requirements.txt`](requirements.txt) — it's Colab-era and outdated; use the repo-level [`env.yml`](../env.yml).

---

Finish line in sight! 🏁

