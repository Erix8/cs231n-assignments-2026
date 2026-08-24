# 🧱 Assignment 2 — ConvNets, Normalization & Captioning

> 🚦 **Status:** ⬜⬜⬜⬜⬜ Not started yet — the deep-learning era begins here. 🌊

Assignment 1 was hand-rolled NumPy; now we go **deep and convolutional** — and meet **PyTorch** for the
first time. The star of the show: understanding *why* BatchNorm and Dropout make training so much
smoother, then watching ConvNets actually work on real images. Oh, and we finish by teaching a model to
describe pictures in words. 🖼️➡️💬

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | `BatchNormalization.ipynb` | BatchNorm forward/backward from scratch (FC + conv), train vs eval mode | ⬜ |
| 2 | `Dropout.ipynb` | Dropout forward/backward from scratch (inverted dropout) | ⬜ |
| 3 | `ConvolutionalNetworks.ipynb` | Conv/pool layers, spatial BatchNorm, group norm, Cython fast layers ⚙️ | ⬜ |
| 4 | `PyTorch.ipynb` | PyTorch warm-up: two-layer net, BatchNorm, Dropout, ConvNet, model visualization | ⬜ |
| 5 | `RNN_Captioning_pytorch.ipynb` | Image captioning with RNN/LSTM on COCO 🖼️➡️💬 | ⬜ |

## 🔑 Ideas I'm chasing

- Normalization: why shifting activations to zero-mean/small-variance tames training 📉
- Dropout as a cheap ensemble / implicit regularizer 🎲
- The conv/pool stack → spatial invariance & parameter sharing 🧱
- Sequence modeling: RNNs/LSTMs and how captions get generated word by word

## 💭 Notes & takeaways

*(To be written as I go — e.g. the moving-average gotcha in BatchNorm eval mode.)*

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| BatchNorm vs no-BatchNorm | BN trains faster + more stable | 🔜 |
| ConvNet on CIFAR-10 | ~75%+ validation accuracy | 🔜 |
| RNN captioning | loss dropping + captions that make sense | 🔜 |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment2 && jupyter notebook`).
3. Run the **first cell** — auto-downloads CIFAR-10 (COCO when you reach the captioning notebook). ⬇️
4. In `ConvolutionalNetworks.ipynb`, build the **Cython** layers once:
   ```bash
   python setup.py build_ext --inplace
   ```
5. Apple Silicon tip: switch the device line to `mps` for a speed boost 🍎⚡

> ⚠️ The captioning notebook trains an RNN — nice on GPU, slow on CPU. If it drags, that's the cue to rent a GPU box. 🚀

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| `cs231n/layers.py` | conv / batch norm / dropout, hand-rolled |
| `cs231n/fast_layers.py` | Cython-backed fast conv/pool ⚡ |
| `cs231n/im2col.py` / `.pyx` | the im2col trick + its fast twin |
| `cs231n/setup.py` | builds the Cython extensions ⚙️ |
| `cs231n/coco_utils.py` | COCO captioning dataset loader |
| `cs231n/rnn_layers_pytorch.py` | RNN/LSTM cells in PyTorch |
| `cs231n/classifiers/fc_net.py` | FC net (A1, now in torch) |
| `cs231n/classifiers/cnn.py` | my ConvNet 🧱 |
| `cs231n/classifiers/rnn_pytorch.py` | the captioning model |
| `collect_submission.ipynb` | 📦 zip + PDF for submission |

---

Let's build a brain! 🧠

