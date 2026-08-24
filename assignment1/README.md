# 🎯 Assignment 1 — Image Classification & Neural Nets

> 🚦 **Status:** ⬜⬜⬜⬜⬜ Not started yet — hoping to tick these off one by one! 🏁

The classic entry point into computer vision. 🖼️ No fancy frameworks, no GPU — just me, NumPy, and a
whole lot of vectorization. By the end I want to *feel* the full image-classification pipeline: from a
lazy kNN that memorizes everything, to linear classifiers that actually learn, all the way up to a
real multi-layer neural net trained with backprop. 💪

## 🗺️ The Journey

| # | Exercise | What I'll implement 🛠️ | Status |
|---|----------|------------------------|--------|
| 1 | `knn.ipynb` | kNN: naive loops → fully vectorized; cross-validate `k` | ⬜ |
| 2 | `softmax.ipynb` | Softmax loss + gradient (vectorized), train on CIFAR-10 | ⬜ |
| 3 | `two_layer_net.ipynb` | Two-layer net forward/backward + gradient check + train | ⬜ |
| 4 | `features.ipynb` | HOG + color-histogram features vs raw pixels 📊 | ⬜ |
| 5 | `FullyConnectedNets.ipynb` | Deep FC nets: affine/ReLU, SGD+Momentum/RMSProp/Adam, dropout, batch/layer norm | ⬜ |

## 🔑 Ideas I'm chasing

- Loss surfaces & the gradient-descent intuition 📉
- Why vectorized code is a lifestyle choice 🏎️
- Backpropagation — from chain rule to `dw`/`dx` in one breath
- Regularization: the quiet hero behind generalization 🛡️

## 💭 Notes & takeaways

*(To be written as I go — mistakes, "aha" moments, and gotchas.)*

## 📊 Scoreboard

| Exercise | Target / result | Got? |
|----------|-----------------|------|
| kNN (k=7) | ~27% test accuracy on CIFAR-10 | 🔜 |
| Softmax | ~35%+ test accuracy | 🔜 |
| Two-layer net | ~48–50% test accuracy | 🔜 |
| Features + linear | ~50%+ (features beat raw pixels!) | 🔜 |
| Best FC net | ≥50% validation accuracy (the assignment bar) | 🔜 |

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

