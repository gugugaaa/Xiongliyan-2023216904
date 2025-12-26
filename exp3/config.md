# 1. 创建环境 (名称替换为你自己的缩写，例如 wjj)
conda create -n xly python=3.9

# 2. 激活环境
conda activate xly

# 3. 安装必要的库
# PyTorch
pip install torch torchvision torchaudio
# OpenCV 和 Matplotlib
pip install opencv-python matplotlib