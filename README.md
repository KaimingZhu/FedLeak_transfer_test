# Integration Notes: Integrating FedLeak (Sec.'25) into Breaching
### ℹ️ **Background & Motivation**

This repository documents how we integrate the official [FedLeak implementation(Zenodo v1)](https://10.5281/zenodo.15532455) into the [Breaching](https://github.com/JonasGeiping/breaching/) evaluation framework, in order to support a unified evaluation protocol across multiple Gradient Inversion Attack baselines.

Our goal is not to modify the core FedLeak algorithm, but to address a practical ***resolution mismatch*** between the two pipelines when evaluating on ImageNet reconstructions.

### 🚩 Resolution Mismatch in Evaluation Pipelines

- **Breaching pipeline**: evaluates reconstruction quality on ImageNet with resolution $224 \times 224 \times 3$.
- **FedLeak generator**: the released generator supports a fixed set of resolutions $\\{32, 64, 128, 256\\}$.

To bridge this gap while preserving the original FedLeak optimization process, we apply a deterministic spatial transformation *after* image generation.

### 🔬 Adaptation Strategy

Following common practices in the vision and GAN literature, we consider two simple and deterministic post-processing variants:

- **Resize-based adaptation**
  - Generator produces $256 \times 256 \times 3$ images
  - Images are resized to $224 \times 224 \times 3$
  - Implementation: [`resize_generator.py`](./resize_generator.py)

- **Center-crop-based adaptation**
  - Generator produces $256 \times 256 \times 3$ images
  - Central $224 \times 224 \times 3$ crop is extracted
  - Implementation: [`center_crop_generator.py`](./center_crop_generator.py)

Both variants keep FedLeak’s optimization objective and training loop unchanged, and are only applied when the target resolution ($224 \times 224 \times 3$) is required by the evaluation pipeline.

We have highlighted our adaptations with `# 🌟 New` and `# 🎯 Origin` in these files.

### 🎯 Sanity Checks

To ensure that these adaptations do not introduce unintended side effects, we perform sanity checks under settings where no resolution adaptation is introduced. In such cases, the adapted generators should behave identically to the original one.

- [`sanity_check_showcase.ipynb`](./sanity_check_showcase.ipynb) Original FedLeak generator (reference behavior)
- [`sanity_check_resize_showcase.ipynb`](./sanity_check_resize_showcase.ipynb) Resize-based generator under sanity-check settings.
- [`sanity_check_centercrop_showcase.ipynb`](./sanity_check_centercrop_showcase.ipynb) Center-crop-based generator under sanity-check settings.

### 🏃 Evaluation at $224 \times 224 \times 3$

We additionally report qualitative reconstructions at $224 \times 224 \times 3$ for both variants:

- [`centercrop_224_showcase.ipynb`](./centercrop_224_showcase.ipynb): the center-crop-based generator.
- [`resize_224_showcase.ipynb`](./resize_224_showcase.ipynb): the resize-based generator.

For transparency, we retain and report both variants in this repository; our experiments use the variant that performs better under identical optimization settings.




