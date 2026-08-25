# 🎯 Assignment 1 — Image Classification & Neural Nets

> 🚦 **Status:** ✅✅✅✅⬜ 4 / 5 done — kNN, Softmax, Two-layer net & Features ticked off! 🏁

The classic entry point into computer vision. 🖼️ No fancy frameworks, no GPU — just me, NumPy, and a
whole lot of vectorization. By the end I want to *feel* the full image-classification pipeline: from a
lazy kNN that memorizes everything, to linear classifiers that actually learn, all the way up to a
real multi-layer neural net trained with backprop. 💪

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | [`knn.ipynb`] | kNN: naive loops → fully vectorized; cross-validate `k` | ✅ |
| 2 | [`softmax.ipynb`] | Softmax loss + gradient (vectorized), train on CIFAR-10 | ✅ |
| 3 | [`two_layer_net.ipynb`] | Two-layer net forward/backward + gradient check + train | ✅ |
| 4 | [`features.ipynb`] | HOG + color-histogram features vs raw pixels 📊 | ✅ |
| 5 | [`FullyConnectedNets.ipynb`] | Deep FC nets: affine/ReLU, SGD+Momentum/RMSProp/Adam, dropout, batch/layer norm | ⬜ |

[`knn.ipynb`]: ./knn.ipynb
[`softmax.ipynb`]: ./softmax.ipynb
[`two_layer_net.ipynb`]: ./two_layer_net.ipynb
[`features.ipynb`]: ./features.ipynb
[`FullyConnectedNets.ipynb`]: ./FullyConnectedNets.ipynb

## 🔑 Ideas I'm chasing

- Loss surfaces & the gradient-descent intuition 📉
- Why vectorized code is a lifestyle choice 🏎️
- Backpropagation — from chain rule to `dw`/`dx` in one breath
- Regularization: the quiet hero behind generalization 🛡️

## 💭 Notes & takeaways

**kNN** ✅
- Double-loop → single-loop → no-loop (matmul + two broadcasts): no-loop is **~140×** faster.
- Bright rows/columns in the distance matrix = images far from *all* of the other set (outliers).
- Cross-validation over `k ∈ {1,3,5,8,10,12,15,20,50,100}` picked **k=10** → 28.2% on test.

**Softmax** ✅
- Gradient is `Xᵀ(p − one_hot)/N + 2·reg·W`; gradcheck passed at ~1e-7 relative error.
- Vectorized loss+grad **~27×** faster than the naive loop.
- Tuned `lr × reg` (num_iters=2000) → best val **37.6%** @ lr=1e-7, reg=1e4; test **35.5%**.
- Learned weights = blurry class-average templates (first-order stats only).

**Two-layer net** ✅
- Modular layers: `affine` / `relu` / `softmax_loss` each with forward+backward; all gradchecks ≤1e-7.
- Architecture: affine→ReLU→affine→softmax, reg loss uses 0.5·λ·ΣW² (Solver handled the SGD loop).
- Default solver (hidden 50, lr 1e-3, 10 epochs) already hit ~51% val — the 36% bar was easy.
- Tuned grid (lr × reg × hidden, 15 epochs) → best val **54.9%** @ lr=1e-3, reg=0.25, hidden=200; test **53.4%**.

**Features** ✅
- Features = HOG (144-d: 9 orientations × 4×4 cells) + hue colour histogram (25-d) → **169-d**, then standardized.
- Softmax on features: best val **52.7%** @ lr=1e-1, reg=1e-3; test **51.0%** (raw-pixel softmax was only 37.6% val → features help a lot!).
- Two-layer net on features: best val **61.6%** @ lr=1.5e-1, reg=1e-4; test **60.0%**.
- Misclassified images are mostly visually similar pairs (plane/ship, car/truck, cat/dog…).

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| kNN (best k=10 via CV) | **28.2%** test accuracy on CIFAR-10 (k=1: 27.4%) | ✅ |
| Softmax | **35.5%** test accuracy (best val **37.6%** @ lr=1e-7, reg=1e4) | ✅ |
| Two-layer net | **53.4%** test accuracy (best val **54.9%** @ lr=1e-3, reg=0.25, hidden=200) | ✅ |
| Features + linear | **51.0%** test (Softmax on features; raw pixels only got 35.5%) | ✅ |
| NN on features | **60.0%** test (best val **61.6%** @ lr=1.5e-1, reg=1e-4) | ✅ |

## 🛩️ Blast off

1. `conda activate cs231n` 🐍
2. Open this folder in VS Code (or `cd assignment1 && jupyter notebook`).
3. Run the **first cell** — it auto-downloads CIFAR-10 (~170 MB). ⬇️
4. Fill in the `TODO`s, run the built-in checks (`rel_error`, accuracy), repeat. 🔁

> 💡 Everything is CPU-only — this one runs on a potato. 🥔 No GPU required!

## 🗂️ Treasure map

| File | What it is |
|------|-----------|
| `cs231n/classifiers/k_nearest_neighbor.py` | my kNN 🗂️ |
| `cs231n/classifiers/linear_classifier.py` | SVM & Softmax live here 📐 |
| `cs231n/classifiers/softmax.py` | softmax, by hand |
| `cs231n/classifiers/fc_net.py` | the deep net 🕸️ |
| `cs231n/layers.py` | affine / ReLU / softmax primitives |
| `cs231n/optim.py` | SGD(+momentum) / RMSProp / Adam |
| `cs231n/solver.py` | the training loop, abstracted |
| `cs231n/features.py` | HOG + color histogram extractors |
| `cs231n/data_utils.py` | CIFAR-10 loader |
| `collect_submission.ipynb` | 📦 zip + PDF for submission |

---

Let's get this bread! 🍞

