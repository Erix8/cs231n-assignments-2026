# 🎯 Assignment 1 — Image Classification & Neural Nets

> 🚦 **Status:** ✅✅✅✅✅ **5 / 5 — Assignment 1 COMPLETE!** 🎉🏁

The classic entry point into computer vision. 🖼️ No fancy frameworks, no GPU — just me, NumPy, and a
whole lot of vectorization. By the end I want to *feel* the full image-classification pipeline: from a
lazy kNN that memorizes everything, to linear classifiers that actually learn, all the way up to a
real multi-layer neural net trained with backprop. 💪

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | [`kNN.ipynb`] | kNN: naive loops → fully vectorized; cross-validate `k` | ✅ |
| 2 | [`Softmax.ipynb`] | Softmax loss + gradient (vectorized), train on CIFAR-10 | ✅ |
| 3 | [`TwoLayerNet.ipynb`] | Two-layer net forward/backward + gradient check + train | ✅ |
| 4 | [`Features.ipynb`] | HOG + color-histogram features vs raw pixels 📊 | ✅ |
| 5 | [`FullyConnectedNets.ipynb`] | Deep FC nets: affine/ReLU, SGD+Momentum/RMSProp/Adam, dropout, batch/layer norm | ✅ |

[`kNN.ipynb`]: ./kNN.ipynb
[`Softmax.ipynb`]: ./Softmax.ipynb
[`TwoLayerNet.ipynb`]: ./TwoLayerNet.ipynb
[`Features.ipynb`]: ./Features.ipynb
[`FullyConnectedNets.ipynb`]: ./FullyConnectedNets.ipynb

## 🔑 Ideas I'm chasing

- Loss surfaces & the gradient-descent intuition 📉
- Why vectorized code is a lifestyle choice 🏎️
- Backpropagation — from chain rule to `dw`/`dx` in one breath
- Regularization: the quiet hero behind generalization 🛡️

## 💭 Notes & takeaways

**Big picture: more capacity → higher ceiling** 📈
kNN ~28% 😴 → softmax ~36% 😐 → 2-layer net ~53% 😎
Each step is a *qualitative* jump, not just a bigger model 🚀

**Memorize ≠ Learn** 🧠
- kNN is lazy AF 🛋️: never trains, just measures distances on the fly (vectorize it → **~140×** faster 🏎️, but still brute force)
- Cross-validating `k` only squeezes it to 28% — no real learning
- A linear classifier does "distill" each class into one template 👻 → but they're just blurry class-averages (first-order stats only)

**Raw pixels are expensive** 💸
- Linear model on raw pixels: ~36%, 'cause it can't decide *what* to look at 🤷
- Same model + hand-crafted HOG/color features → **~51%** ✨ the features do the heavy lifting, not the model
- Let the *net* decide → a single hidden layer on raw pixels ~53%, already beats "features + linear" 🎉
- Learnt + hand-crafted features together → **~60%** 🏆
- TL;DR: feature *learning* > feature *engineering* 💡

**Craft tips** 🔧
- Vectorization is a lifestyle 🏎️ (kNN ~140×, softmax ~27× — usually clearer too)
- Backprop is modular & reliable: affine/relu/softmax each carry forward+backward, just stack 'em, gradcheck stays ≤1e-7 ✅

**Deep nets flip the tuning game** 🎛️
- Only 50 tiny images? 100% in 20 epochs 😅 — fitting was never the problem
- Deeper = way more sensitive to `weight_scale` (activations/gradients compound per layer) ⚠️
- "Just pick a good lr" ain't enough: momentum / RMSProp / Adam each fix different issues 💊
- Best: `[200,200]` + Adam + sane ws/reg → **56.6% val**, ~53% test 🎯 (clears the 50% bar)

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| kNN (best k=10 via CV) | **28.2%** test accuracy on CIFAR-10 (k=1: 27.4%) | ✅ |
| Softmax | **35.5%** test accuracy (best val **37.6%** @ lr=1e-7, reg=1e4) | ✅ |
| Two-layer net | **53.4%** test accuracy (best val **54.9%** @ lr=1e-3, reg=0.25, hidden=200) | ✅ |
| Features + linear | **51.0%** test (Softmax on features; raw pixels only got 35.5%) | ✅ |
| NN on features | **60.0%** test (best val **61.6%** @ lr=1.5e-1, reg=1e-4) | ✅ |
| Best FC net | **52.6%** test / **56.6%** val @ `[200,200]`, Adam, reg 0.01 (bar: 50%) | ✅ |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment1 && jupyter notebook`).
3. Run the **first cell** — it auto-downloads CIFAR-10 (~170 MB). ⬇️
4. Fill in the `TODO`s, run the built-in checks (`rel_error`, accuracy), repeat. 🔁

> 💡 Everything is CPU-only — this one runs on a potato. 🥔 No GPU required!

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| [`cs231n/classifiers/k_nearest_neighbor.py`] | my kNN 🗂️ |
| [`cs231n/classifiers/linear_classifier.py`] | SVM & Softmax live here 📐 |
| [`cs231n/classifiers/softmax.py`] | softmax, by hand |
| [`cs231n/classifiers/fc_net.py`] | the deep net 🕸️ |
| [`cs231n/layers.py`] | affine / ReLU / softmax primitives |
| [`cs231n/optim.py`] | SGD(+momentum) / RMSProp / Adam |
| [`cs231n/solver.py`] | the training loop, abstracted |
| [`cs231n/features.py`] | HOG + color histogram extractors |
| [`cs231n/data_utils.py`] | CIFAR-10 loader |
| [`collect_submission.ipynb`] | 📦 zip + PDF for submission |

[`cs231n/classifiers/k_nearest_neighbor.py`]: ./cs231n/classifiers/k_nearest_neighbor.py
[`cs231n/classifiers/linear_classifier.py`]: ./cs231n/classifiers/linear_classifier.py
[`cs231n/classifiers/softmax.py`]: ./cs231n/classifiers/softmax.py
[`cs231n/classifiers/fc_net.py`]: ./cs231n/classifiers/fc_net.py
[`cs231n/layers.py`]: ./cs231n/layers.py
[`cs231n/optim.py`]: ./cs231n/optim.py
[`cs231n/solver.py`]: ./cs231n/solver.py
[`cs231n/features.py`]: ./cs231n/features.py
[`cs231n/data_utils.py`]: ./cs231n/data_utils.py
[`collect_submission.ipynb`]: ./collect_submission.ipynb

---

Let's get this bread! 🍞

