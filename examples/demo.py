#%% import
from __future__ import division

from utils import stain_utils as utils
from models import stainNorm_Vahadane, stainNorm_Reinhard, stainNorm_Macenko

import numpy as np

#%% defines
# normalization method (Reinhard [0], Macenko [1], Vahadane [2])
normalization_method = 0
input_dir = './data/'
output_dir = './results/'

#%% load data
for img in range(4):
    img_target = utils.read_image(input_dir + "/normal_real_"+ str(img) +".png")
    img_source_1 = utils.read_image(input_dir + "/short_real_"+ str(img) +".png")
    img_source_2 = utils.read_image(input_dir + "/long_real_"+ str(img) +".png")
    img_source_3 = utils.read_image(input_dir + "/onlyH_real_"+ str(img) +".png")
    img_source_4 = utils.read_image(input_dir + "/onlyE_real_"+ str(img) +".png")

    #%% normalize
    if normalization_method == 0:
        # Reinhard
        method = 'Reinhard'
        normalizer= stainNorm_Reinhard.Normalizer()
    elif normalization_method == 1:
        # Macenko
        method = 'Macenko'
        normalizer = stainNorm_Macenko.Normalizer()
    elif normalization_method == 2:
        # Vahadane
        method = 'Vahadane'
        normalizer = stainNorm_Vahadane.Normalizer()
    else:
        print('enter valid normalization method (Reinhard [0], Macenko [1], Vahadane [2])')
        exit()

    normalizer.fit(img_target)
    normalized=utils.build_stack((img_target, img_source_1, img_source_2, img_source_3, img_source_4, np.ones((512,512,3)) , normalizer.transform(img_source_1), normalizer.transform(img_source_2), normalizer.transform(img_source_3), normalizer.transform(img_source_4)))
    utils.patch_grid(normalized,width=5,save_name=output_dir + method + '_' + str(img))


