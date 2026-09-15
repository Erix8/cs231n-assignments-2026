# 🚀 CS231N Assignments — Spring 2026 · My Solutions 📝

My self-study playground for Stanford's **CS231n: Deep Learning for Computer Vision** 👀. 

This repo is **local-first & Colab-free** — every notebook is patched to run straight on your own machine, and my solutions are shared here as a learning record. 🎉 

For official course stuff (schedule, lectures, assignment overviews), check out the [cs231n website].

[cs231n website]: https://cs231n.stanford.edu/2026

## 🗂️ Assignment Contents

| # | Assignment | Progress |
|---|------------|----------|
| 1 | [Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets 🔍](./assignment1/README.md) | ✅ **Completed** — 5/5 notebooks (hand-rolled NumPy: kNN → softmax → 2-layer net → deep FC nets, **56.6%** val 🏆) |
| 2 | [Batch Normalization, Dropout, Convolutional Nets, Network Visualization, Image Captioning with RNNs 🧱](./assignment2/README.md) | ✅ **Completed** — 5/5 notebooks (hand-rolled BN/dropout/convs → PyTorch **88.8%** val 🏆) |
| 3 | [Image Captioning with Transformers, Self-Supervised Learning, Diffusion Models, CLIP and DINO Models ✨](./assignment3/README.md) | 🟨 **In progress** — 3/4 notebooks + 1 code-complete (attention → ViT → diffusion → CLIP/DINO, **0.645** mean IoU one-shot segmentation 🏆; SimCLR training pending ⏳) |

> 📖 Click any assignment name to open its README & directory map.

## 💡 Quick Start

```bash
conda env create -f env.yml
conda activate cs231n
```

TL;DR: create & activate the env, then get coding! 🐍

### 🏃 How to run a notebook

1. Open the **assignment folder** (e.g. `assignment1/`) in VS Code — or `cd assignment1 && jupyter notebook` from the terminal.
2. Run the **first cell** of any notebook: it auto-locates the `cs231n` package and downloads the dataset if missing. ⬇️
3. Code, run, repeat. 💪

> ⚠️ Tip: open the assignment folder itself as your workspace — the setup cell will throw a friendly error if it can't find `cs231n`.

### 📦 Environment notes

- **PyTorch 2.13** with **MPS** support on Apple Silicon 🍎⚡ (auto-falls back to CPU elsewhere)
- **`imageio<3`** is pinned — the assignment code uses `from imageio import imread`, which was removed in imageio 3.x
- **`tensorflow` + `tensorflow-datasets`** are only needed for the CLIP/DINO notebook (DAVIS video) — comment them out in `env.yml` if you don't need them

## 🤝 Study & Sharing Notes

- My solutions live in this repo for self-study reference — **write your own code first**, then peek if you get stuck! ✍️
- Every exercise ships with built-in sanity checks (`rel_error`, accuracy tests), so you'll know instantly whether your implementation is correct ✅

Happy learning! 🎓
