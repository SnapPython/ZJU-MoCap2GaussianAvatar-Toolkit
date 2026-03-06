# ZJU-MoCap2GaussianAvatar-Toolkit

本项目是一个专为 **GaussianAvatar (CVPR 2024)** 等基于3DGS的人体重建模型开发的数据预处理工具。
作为作者本科毕业设计的产出部分，它能够将 **ZJU-MoCap** 数据集的原始格式一键转换为兼容 InstantAvatar 风格的 `npz` 驱动格式。

---

## 📂 项目文件结构
```
ZJU-MoCap2GaussianAvatar-Toolkit/
├── environment.yml          # Conda 环境一键配置文件
├── LICENSE                  
├── README.md                # 项目说明文档
├── main.py                  # 核心数据转换脚本
├── .gitignore               
└── data/                    # 【本地数据目录】
    ├── raw/                 # 放入解压后的原始数据
    └── prepared/            # 脚本运行后的标准化输出结果
```
---

## 🔄 数据映射关系
本工具自动将 ZJU 的 `annots.npy` 结构重新映射为以下格式：

| 原始 ZJU 数据 (`annots.npy`) | 转换后 (GaussianAvatar 格式) | 说明 |
| --- | --- | --- |
| `cams['K']` | `intrinsics` (in `cameras.npz`) | 相机内参矩阵 |
| `cams['R']`, `cams['T']` | `extrinsics` (in `cameras.npz`) | W2C 外参矩阵 (已转为米) |
| `poses` + `Rh` | `poses` (in `poses_optimized.npz`) | 融合后的 72 维姿态参数 |
| `Th` | `trans` (in `poses_optimized.npz`) | 全局平移向量 (已转为米) |
| `shapes` | `betas` (in `poses_optimized.npz`) | SMPL 形态参数 (10维) |

---

## 🛠️ 快速上手

### 1. 克隆本仓库

```
git clone https://github.com/SnapPython/ZJU-MoCap2GaussianAvatar-Toolkit.git
cd ZJU-MoCap2GaussianAvatar-Toolkit
```

### 2. 配置环境
```
conda env create -f environment.yml
conda activate zju2gs
```

### 3. 准备数据
1. 确保你已获取 ZJU-MoCap 数据集（`CoreView_xxx`）。
2. 并将解压后的文件夹移动到 data/raw 下。

### 4. 运行一键转换

运行以下命令，将特定视角的序列转换为 GaussianAvatar 格式：

```
python main.py --raw_dir ./data/raw --out_dir ./data/prepared --view 0
```

转换完成后，`data/prepared/` 下生成包含 images/, masks/, cameras.npz, poses_optimized.npz 的文件夹。

在 GaussianAvatar 仓库中，利用作者提供的工具完成后续格式转换。

---

## 🎓 引用

如果你在毕设或研究中使用了此脚本或相关数据集，请引用以下原始论文：

```bibtex
@inproceedings{hu2024gaussianavatar,
  title={GaussianAvatar: Towards 3D Gaussian Splatting for Realistic Human Avatar Modeling},
  author={Hu, Liangxiao and others},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{peng2021neural,
  title={Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans},
  author={Peng, Sida and others},
  booktitle={CVPR},
  year={2021}
}
```

## 📜 许可证

本项目基于 **MIT License** 开源。

---
