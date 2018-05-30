curl "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh

conda install -y pytorch torchvision cuda91 -c pytorch
conda install -y numpy ipython jupyter tensorflow pillow cython scipy requests cupy
pip install tensorboardX tqdm git+https://github.com/AMLab-Amsterdam/lie_learn \
    git+https://github.com/pimdh/svae-temp.git git+https://github.com/jonas-koehler/s2cnn.git pynvrtc
