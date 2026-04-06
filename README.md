# DUHA-Net: Dual-branch Uncertainty-aware Haar-enhanced Aggregation Network
###  🔒 Review Status

This work is currently under review at **Pattern Analysis and Applications**. 
The code is made available to facilitate the review process.

### 📖 Introduction

DUHA-Net is a novel semantic segmentation framework for remote sensing imagery that addresses challenges of spatial heterogeneity, inter-class confusion, and ambiguous boundaries. The main contributions are:

1. **Dual-Branch Architecture**: Synergistically integrates ConvNeXt (local features) and Swin Transformer (global dependencies) for complementary representations.

2. **Haar Enhancement Module (HEM)**: Frequency-domain enhancement via Haar wavelet transform to suppress noise and reinforce structural edges.

3. **Semantic-Guided Decoupling Aggregation (S-GDA)**: Class-aware attention mechanism that explicitly decouples semantic features from heterogeneous branches.

4. **Uncertainty-aware Dual-branch Training (UDT)**: Adaptive training strategy that quantifies spatial uncertainty and weights challenging samples through dynamic thresholding.

### 🚀 **Requirements**

#### **Environment**

* Python >= 3.8
* PyTorch >= 1.10.0
* CUDA >= 11.1 (recommended)
* Linux / Windows / macOS
* **Hardware:** NVIDIA RTX A4000 (16GB VRAM) - Training uses batch size 4 with 256×256 input resolution, consuming approximately 10-12GB GPU memory.
### **📦 Installation**

#### Step 1: Create virtual environment (recommended)
`conda create -n duhanet python=3.10`

`conda activate duhanet`

#### Step 2: Install PyTorch with CUDA 11.8
`pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118`

#### Step 3: Install other dependencies
`pip install -r requirements.txt`

### 📊 Dataset Preparation

The ISPRS Vaihingen and Potsdam datasets are publicly available through the ISPRS 2D Semantic Labeling Benchmark.

**How to access:**

Visit the ISPRS website: https://www.isprs.org

Navigate to "Technical Commissions" → "Benchmarks"

Find the "2D Semantic Labeling Contest" or "Urban Semantic Labeling" section

Follow the links to download the Vaihingen and Potsdam datasets
* data/
* ├── vaihingen/
* │   ├── train/
* │   │   ├── images/
* │   │   └── labels/
* │   ├── val/
* │   │   ├── images/
* │   │   └── labels/
* │   ├── test/
* │   │   ├── images/
* │   │   └── labels/  

### 🚀 Training&Testing
`python train.py --data_name vaihingen --epoch 100 --batchsize 4 --consistency_weight 0.4 --entropy_weight 0.4 --hfem_weight 0.2 --uncertainty_loss_weight 0.2 --consistency_loss_weight 0.1 --base_threshold 0.7 --warmup_epochs 5 --weight_type exp`


`python test.py --model_path ./checkpoints/xxx_vaihingen_best.pth --data_name vaihingen --batchsize 4 --trainsize 256 --use_hfem_stages 0,1 --gpu_id 0 --save_path ./test_results/`

