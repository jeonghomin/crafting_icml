# Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution (Implementation)

**Ruofan Zhang, Jinjin Gu, Haoyu Chen, Chao Dong, Yulun Zhang, Wenming Yang**  
(ICML 2023 Accepted Paper)

[arXiv:2305.18107](https://arxiv.org/abs/2305.18107)

---

## Overview

Super-resolution (SR) techniques designed for real-world applications commonly encounter two primary challenges: **generalization performance** and **restoration accuracy**.  
This repository contains an implementation of the method proposed in the ICML 2023 paper:

> **Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution**

We demonstrate that when methods are trained using complex, large-range degradations to enhance generalization, a decline in accuracy is inevitable. However, since the degradation in a certain real-world application typically exhibits a limited variation range, it becomes feasible to strike a trade-off between generalization performance and testing accuracy within this scope.

---

## Key Features

- Novel degradation distribution crafting using a small set of reference images
- Binned representation of the degradation space
- Fréchet distance between degradation distributions for optimization
- Achieves a balance between generalization and accuracy in real-world SR tasks

---

## Getting Started

### Requirements

- Python >= 3.7

### Installation

```bash
git clone https://github.com/jeonghomin/CTD.git
cd CTD
pip install -r requirements.txt
```

### Usage

```bash
# Example: Training and testing scripts
python train.py --config configs/your_config.yaml
python test.py --model_path path/to/model.pth
```

(Add detailed usage, options, and dataset preparation instructions as appropriate for your implementation.)

---

## Results

- The proposed method significantly improves the performance of test images while preserving generalization capabilities in real-world applications.
- (Add performance tables, example images, or graphs as needed.)

---

## Citation

If you use this code or paper, please cite as below:

```bibtex
@inproceedings{zhang2023crafting,
  title={Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution},
  author={Zhang, Ruofan and Gu, Jinjin and Chen, Haoyu and Dong, Chao and Zhang, Yulun and Yang, Wenming},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

---

## License

(Add your license information, e.g., MIT, Apache-2.0, etc.)

---

## Contact
- Email: [jeongho.min@unist.ac.kr](mailto:jeongho.min@unist.ac.kr)
- Issues and PRs are welcome!

