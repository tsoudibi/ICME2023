# ICME2023

![](animation.gif)

http://bdd-data.berkeley.edu/

Trained U-Net based Keras model on Drivable Map dataset.

modified from https://github.com/YoongiKim/RoadDetector

# 環境建立：
'''
conda create --n tf python=3.9 -c conda-forge cudatoolkit=11.6 cudnn=8.8

# 重新登出再登入SSH

conda activate tf
python3 -m pip install tensorflow

# 驗證安裝
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 套件安裝
pip install scikit-learn matplotlib
'''
