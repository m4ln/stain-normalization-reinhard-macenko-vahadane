Forked from https://github.com/wanghao14/Stain_Normalization

# Overview

Implementation of a few common stain normalization techniques ([Reinhard](http://ieeexplore.ieee.org/document/946629/), [Macenko](http://ieeexplore.ieee.org/document/5193250/), [Vahadane](http://ieeexplore.ieee.org/document/7164042/)) in Python (3.5).

For usage see the notebook ```demo.ipynb``` or ```demo.py```

In short do something like (all techniques have the same API, where we create a stain normalization object or *Normalizer*. The fit and transform methods are then the most important).

```
n = stainNorm_Reinhard.Normalizer()
n.fit(target_image)
out = n.transform(source_image)
```

If you want Hematoxylin do something like

```
n = stainNorm_Vahadane.Normalizer()
out = n.hematoxylin(source_image)
```

We can also view the stains seperated by e.g.

```
n = stainNorm_Vahadane.Normalizer()
n.fit(target_image)
stain_utils.show_colors(n.target_stains())
```

We use the [SPAMS](http://spams-devel.gforge.inria.fr/index.html) (SPArse Modeling Software) package. Use with Python via e.g https://anaconda.org/conda-forge/python-spams

Below we show the application of the techniques to a few images (in data folder). We normalize to the first image and for Macenko and Vahadane also show the extracted Hematoxylin channel. Below that are a few more challenging images (also in data folder). All images are taken from the [ICIAR 2018 challenge](https://iciar2018-challenge.grand-challenge.org/).

One change to the vanilla methods is used. With all images we first apply a brightness standardizing step (below). This is especially useful in handling the more challenging images (which are typically too dim) and does not damage performance for the other images. 

```
def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)
```
