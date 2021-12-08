#%% import
import numpy as np
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
import cv2 as cv

#%% utils
def read_image(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im

# visualization color map
vals = np.random.rand(256, 3)
vals[0, ...] = [0.9, 0.9, 0.9]
cMap = ListedColormap(1 - vals)

# for visualization
ymin, ymax, xmin, xmax = 1000, 1500, 2500, 3000

# for reproducibility
np.random.seed(0)

#%% read images
# Read the images
tissue_rgb = read_image("./data/HE/HE[x=192,y=29376,w=256,h=256].tif")
i2 = read_image("./data/longHE/longHE[x=4224,y=30912,w=256,h=256].tif")
i3 = read_image("./data/onlyE/onlyE[x=768,y=29376,w=256,h=256].tif")
i4 = read_image("./data/onlyH/onlyH[x=2112,y=30912,w=256,h=256].tif")
i5 = read_image("./data/shortHE/shortHE[x=1536,y=29952,w=256,h=256].tif")

#%%
# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

# TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
# for macenco (obtained using rgb_separate_stains_macenko_pca()
# and reordered such that columns are the order:
# Hamtoxylin, Eosin, Null
W_target = np.array([
    [0.5807549,  0.08314027,  0.08213795],
    [0.71681094,  0.90081588,  0.41999816],
    [0.38588316,  0.42616716, -0.90380025]
])

#%%
# get mask of things to ignore
mask_out, _ = get_tissue_mask(
    tissue_rgb, deconvolve_first=True,
    n_thresholding_steps=1, sigma=1.5, min_size=30)
mask_out = resize(
    mask_out == 0, output_shape=tissue_rgb.shape[:2],
    order=0, preserve_range=True) == 1
#%%
# Let’s visualize the data
f, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(tissue_rgb)
ax[1].imshow(mask_out)
# ax[1].imshow(mask_out[ymin:ymax, xmin:xmax], cmap=cMap)
plt.show()

#%%
# Reinhard normalization - without masking
tissue_rgb_normalized = reinhard(
    tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])

def vis_result():
    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(tissue_rgb)
    ax[1].imshow(tissue_rgb_normalized)
    plt.show()

vis_result()

# Reinhard normalization - with masking
tissue_rgb_normalized = reinhard(
    tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'],
    mask_out=mask_out)
vis_result()

#%%
# Deconvolution-based normalization
print(deconvolution_based_normalization.__doc__)
print(color_deconvolution_routine.__doc__)
print(stain_unmixing_routine.__doc__)

# Macenko normalization - without masking
stain_unmixing_routine_params = {
    'stains': ['hematoxylin', 'eosin'],
    'stain_unmixing_method': 'macenko_pca',
}
tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
vis_result()

# Macenko normalization - with masking
tissue_rgb_normalized = deconvolution_based_normalization(
        tissue_rgb,  W_target=W_target,
        stain_unmixing_routine_params=stain_unmixing_routine_params,
        mask_out=mask_out)
vis_result()

# “Smart” color augmentation
print(perturb_stain_concentration.__doc__)
print(rgb_perturb_stain_concentration.__doc__)


# Let’s perturb the H&E concentrations a bit
rgb = tissue_rgb[ymin:ymax, xmin:xmax, :]
exclude = mask_out[ymin:ymax, xmin:xmax]
augmented_rgb = rgb_perturb_stain_concentration(rgb, mask_out=exclude)
def vis_augmentation():
    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(rgb)
    ax[1].imshow(augmented_rgb)
    plt.show()

vis_augmentation()

# Try a few more times
for _ in range(5):
    augmented_rgb = rgb_perturb_stain_concentration(rgb, mask_out=exclude)
    vis_augmentation()
