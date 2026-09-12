# 🧱 Assignment 2 — ConvNets, Normalization & Captioning

> 🚦 **Status:** ✅✅✅✅✅ **5 / 5 — Assignment 2 COMPLETE!** 🎉🏁

Assignment 1 was hand-rolled NumPy; now we go **deep and convolutional** — and meet **PyTorch** for the
first time. The star of the show: understanding *why* BatchNorm and Dropout make training so much
smoother, then watching ConvNets actually work on real images. Oh, and we finish by teaching a model to
describe pictures in words. 🖼️➡️💬

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | [`BatchNormalization.ipynb`](BatchNormalization.ipynb) | BatchNorm forward/backward from scratch (FC + conv), train vs eval mode | ✅ |
| 2 | [`Dropout.ipynb`](Dropout.ipynb) | Dropout forward/backward from scratch (inverted dropout) | ✅ |
| 3 | [`ConvolutionalNetworks.ipynb`](ConvolutionalNetworks.ipynb) | Conv/pool layers, spatial BatchNorm, group norm, Cython fast layers ⚙️ | ✅ |
| 4 | [`PyTorch.ipynb`](PyTorch.ipynb) | PyTorch warm-up: two-layer net, BatchNorm, Dropout, ConvNet, model visualization | ✅ |
| 5 | [`RNNCaptioning.ipynb`](RNNCaptioning.ipynb) | Image captioning with RNN/LSTM on COCO 🖼️➡️💬 | ✅ |

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
| BatchNorm vs no-BatchNorm | train 0.79 vs 0.71 @ 10 epochs, and far more robust to weight-init scale | ✅ |
| Dropout as regularizer | smaller train/val gap — val 0.31 vs 0.26, train 0.90 vs 0.92 | ✅ |
| Naive conv/pool vs Cython fast layers | fast clearly wins — 427× conv fwd, 765× conv bwd, ~56× pool | ✅ |
| ConvNet on CIFAR-10 | 47.6% train / 49.9% val after 1 epoch | ✅ |
| PyTorch barebones / Module / Sequential nets | beat the Part II–IV bars 43 / 49 / 47 / 57% val | ✅ |
| PyTorch CIFAR-10 challenge | 88.8% val / 87.4% test within 10 epochs | ✅ |
| RNN captioning | overfit 50 imgs to loss 0.013; train captions verbatim, val gibberish | ✅ |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment2 && jupyter notebook`).
3. Run the **first cell** — auto-downloads CIFAR-10 (COCO when you reach the captioning notebook). ⬇️
4. In [`ConvolutionalNetworks.ipynb`](ConvolutionalNetworks.ipynb), build the **Cython** layers once:
   ```bash
   python setup.py build_ext --inplace
   ```
5. Apple Silicon tip: switch the device line to `mps` for a speed boost 🍎⚡

> ⚠️ The captioning notebook trains an RNN — nice on GPU, slow on CPU. If it drags, that's the cue to rent a GPU box. 🚀

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| [`cs231n/layers.py`](cs231n/layers.py) | conv / batch norm / dropout, hand-rolled |
| [`cs231n/fast_layers.py`](cs231n/fast_layers.py) | Cython-backed fast conv/pool ⚡ |
| [`cs231n/im2col.py`](cs231n/im2col.py) / [`cs231n/im2col_cython.pyx`](cs231n/im2col_cython.pyx) | the im2col trick + its fast twin |
| [`cs231n/setup.py`](cs231n/setup.py) | builds the Cython extensions ⚙️ |
| [`cs231n/coco_utils.py`](cs231n/coco_utils.py) | COCO captioning dataset loader |
| [`cs231n/rnn_layers_pytorch.py`](cs231n/rnn_layers_pytorch.py) | RNN/LSTM cells in PyTorch |
| [`cs231n/classifiers/fc_net.py`](cs231n/classifiers/fc_net.py) | FC net (A1, now in torch) |
| [`cs231n/classifiers/cnn.py`](cs231n/classifiers/cnn.py) | my ConvNet 🧱 |
| [`cs231n/classifiers/rnn_pytorch.py`](cs231n/classifiers/rnn_pytorch.py) | the captioning model |
| [`collect_submission.ipynb`](collect_submission.ipynb) | 📦 zip + PDF for submission |

---

Let's build a brain! 🧠

