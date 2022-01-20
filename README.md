# Code for "Explorations in Homeomorphic Variational Auto-Encoding"
by Luca Falorsi*, Pim de Haan*, Tim R. Davidson*, Nicola De Cao, Maurice Weiler, Patrick Forré and Taco S. Cohen.

[Link to paper](https://arxiv.org/abs/1807.04689)
[Animated results](https://sites.google.com/view/lie-vae)


The relevant code to implement SO(3) valued latent variables can be found in the `lie_vae` package. The code needed to reproduce the experiments can be found in `lie_vae.experiments`.

For questions file an issue or email to either:
- Luca Falorsi - <luca.falorsi@gmail.com>
- Pim de Haan - <pimdehaan@gmail.com>


## Dependencies

```
conda install -y pytorch torchvision cuda91 -c pytorch
conda install -y numpy ipython jupyter tensorflow pillow cython scipy requests
pip install tensorboardX tqdm git+https://github.com/AMLab-Amsterdam/lie_learn \
    git+https://github.com/pimdh/svae-temp.git 
```

The sphere cube data can be generated with the `python -m lie_vae.experiments.gen_spherecube_pairs` (see file for details, this requires having installed Blender 2.79b) or for limited time be downloaded [here](https://drive.google.com/file/d/1pZf4_B__XtL6DujHIhuARtYQk-JumZin/view?usp=sharing).
