# SFTGNN
Official code repository of paper "Graph Neural Network for Crystal Property Prediction Based on Spherical Fourier Transform" by Zhen Jiang and Hua Tian.

<br><img src="images/TOC.svg" alt="MP Dataset" width="500"/><br>
<br><img src="images/Model.png" alt="MP Dataset" width="500"/><br>
### Benchmarked results
- **The Material Project Dataset**
<br><img src="images/table3_MP.jpeg" alt="MP Dataset" width="500"/><br>
- **Jarvis Dataset**
<br><img src="images/table4_Jarvis.jpeg" alt="Jarvis Dataset" width="500"/><br>
- **Efficiency**
<br><img src="images/table5_Efficiency.jpeg" alt="Model Efficiency" width="500"/><br>
## Environment Setup

We use **conda** to manage dependencies and set up the runtime environment. The recommended versions are:

- **Python**: 3.11.8  
- **PyTorch**: 2.2.2  
- **CUDA**: 12.1  

### 1. Create and activate a conda environment

```bash
conda create --name sftgnn python=3.11.8
conda activate sftgnn
```
### 2. Install PyTorch and CUDA dependencies

```bash
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### 3.Install PyTorch Geometric and related packages
We use PyTorch Geometric for implementing our graph neural network. Install it and related components using the following commands:
```bash
pip install torch_geometric==2.5.2
pip install torch-cluster==1.6.3 torch-scatter==2.1.2 torch-sparse==0.6.18 torch-spline-conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-2.2.2%2Bcu121.html
```
### 4. Install additional dependencies
After completing the core setup, install other required packages via:
```bash
pip install -r requirements.txt
```

## Availability
🔒 The full implementation of this project is currently under review as part of an academic publication. 
The source code, pretrained models, and additional resources will be made publicly available upon acceptance.
