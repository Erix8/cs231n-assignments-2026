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

**BatchNorm: speed is the headline, robustness is the story** 🎛️
- Trains faster (train 0.79 vs 0.71 @ 10 epochs) — but the real win is weight-init robustness: the baseline only learns in a narrow `weight_scale ≈ 1e-2..5e-2` window, BN works from ~1e-2 all the way to ~1e0 🛡️
- The eval-mode gotcha ⚠️: training normalizes with *batch* stats, testing with the *running averages* — that's the whole reason `running_mean`/`running_var` exist, and forgetting them quietly tanks accuracy
- It's batch-size dependent: tiny batches give noisy mean/var → bs 5 < 10 < 50 📉
- LayerNorm is the batch-free fix (normalize per data point, train ≡ test) — but it dies when the *feature* dim is tiny: same failure mode, other axis 🔁

**Dropout: a cheap ensemble, not free** 🎲
- Inverted dropout's `/p` keeps the expected activation scale identical at train and test — drop it and test-time activations are `1/p` too big
- 500 imgs: no-dropout train **0.92** / val **0.26** → dropout(0.25) train **0.90** / val **0.31** → smaller train/val gap, slightly lower train acc 🛡️
- It buys generalization by spending training accuracy — worth it only when you're actually overfitting

**Convs: identical math, ~500× the speed** ⚡
- Naive conv/pool are just nested loops; Cython im2col gives **427×** (conv fwd), **765×** (conv bwd) and **~56×** (pooling) 🏎️
- "Spatial" BN is literally vanilla BN with the channel axis moved last; group norm is LayerNorm over a `(N·G, C/G·H·W)` reshape — same formula, different fold

**PyTorch: the abstraction ladder** 🪜
- barebones tensors (43% / 49%) → `nn.Module` (47%) → `nn.Sequential` (57%): same net, less code on every rung
- Part V big win 🏆: VGG-style + BN + **global-average-pool** (ditch the giant FC) + dropout + augmentation + cosine-annealed Adam → **88.8% val / 87.4% test in 10 epochs** (bar was 70%)
- BN + GAP + a sane LR schedule did more than any amount of architecture fiddling

**RNNs: overfit in a blink, generalize never** 🖼️➡️💬
- 50 images → loss **80 → 0.013** in 100 iterations; train captions come back *verbatim*, val captions are word salad 🥗 — exactly the memorization the notebook predicts
- My LSTM matched `torch.nn.LSTM` to **3.8e-16** once I mapped PyTorch's `i,f,g,o` gate order onto cs231n's `i,f,o,g` 🎯
- Word-level vs char-level: a few dozen tokens (tiny vocab, open vocabulary, no `<UNK>`) vs much longer sequences that are harder to learn 📝

**Craft tips** 🔧
- Gradient-check every hand-written layer before trusting a training run — everything here stayed ≤1e-6 ✅
- A captioning model behaves differently at train vs test: feed ground truth vs sample your own words
- ~17% of the 2014 Flickr URLs are dead 🖼️ — guard `image_from_url` with a `None` check before `imshow`

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
5. Device is auto-detected: the PyTorch notebook uses `mps` when it's available and falls back to CPU otherwise — nothing to configure 🍎⚡

> 💻 Everything here ran **locally, with no GPU rental** — just hit run. The captioning notebook trains its RNN in a couple of minutes. 🚀

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

