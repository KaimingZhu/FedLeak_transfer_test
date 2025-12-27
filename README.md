This repository provides the implementation of FedLeak, transferred from [Zenodo(Version v1)](10.5281/zenodo.15532455).

Run FedLeak:
```bash
python attack.py
```

---

#### Datasets

The repository contains 16 images from ImageNet for testing.

##### Datasets Available via Torchvision
- MNIST
- SVHN
- CIFAR-10
- CIFAR-100
- ImageNet

##### Medical Datasets
- **HAM10000** [Download Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
- **Lung & Colon Cancer Histopathological Images** [Download Link](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---

#### My Experimental Environment

| Component | Version/Config |
|-----------|----------------|
| Python    | 3.11.5         |
| GPU       | NVIDIA 4090    |
| CUDA      | 12.2           |
| OS        | CentOS 7       |
| dependencies | requirements.txt |

---

#### Acknowledgements

This repository references the following repositories:
1. https://github.com/mit-han-lab/dlg
2. https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients
3. https://github.com/czhang024/CI-Net