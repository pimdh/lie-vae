FROM nvidia/cuda:9.1-base-ubuntu16.04

RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -y curl git build-essential
RUN curl "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /root/miniconda3
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN /bin/bash -c "source activate"
RUN conda install -y pytorch torchvision cuda91 -c pytorch
RUN conda install -y numpy ipython jupyter tensorflow pillow cython scipy requests cupy
RUN pip install tensorboardX tqdm git+https://github.com/AMLab-Amsterdam/lie_learn \
    git+https://github.com/pimdh/svae-temp.git git+https://github.com/jonas-koehler/s2cnn.git pynvrtc

EXPOSE 8888
EXPOSE 6006

WORKDIR /app
