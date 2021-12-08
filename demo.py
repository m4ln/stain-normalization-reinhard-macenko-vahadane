#%% import
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane

import numpy as np
import matplotlib.pyplot as plt

#%% load data
input_dir = '/home/mr38/sds_hd/sd18a006/marlen/datasets/stainNormalization/HEV/H.18.4262/stainTools/'
i1 = utils.read_image(input_dir + "/HE/HE[x=192,y=29376,w=256,h=256].tif")
i2 = utils.read_image(input_dir + "/longHE/longHE[x=4224,y=30912,w=256,h=256].tif")
i3 = utils.read_image(input_dir + "/onlyE/onlyE[x=768,y=29376,w=256,h=256].tif")
i4 = utils.read_image(input_dir + "/onlyH/onlyH[x=2112,y=30912,w=256,h=256].tif")
i5 = utils.read_image(input_dir + "/shortHE/shortHE[x=1536,y=29952,w=256,h=256].tif")

#%% print data
output_dir = './results'
stack=utils.build_stack((i1,i2,i3,i4,i5))
utils.patch_grid(stack,width=3,save_name=output_dir + '/original.pdf')

#%% Reinhard normalization
n=stainNorm_Reinhard.Normalizer()
n.fit(i1)
normalized=utils.build_stack((i1,n.transform(i2),n.transform(i3),n.transform(i4),n.transform(i5)))

#%% print
utils.patch_grid(normalized,width=3,save_name='Reinhard.pdf')

#%% Macenko normalization
n=stainNorm_Macenko.Normalizer()
n.fit(i1)
normalized=utils.build_stack((i1,n.transform(i2),n.transform(i3),n.transform(i4),n.transform(i5)))

#%% print
utils.patch_grid(normalized,width=3,save_name=output_dir + '/Macenko.pdf')

#%% Vahadane normalization
n=stainNorm_Vahadane.Normalizer()
n.fit(i1)
normalized=utils.build_stack((i1,n.transform(i2),n.transform(i3),n.transform(i4),n.transform(i5)))

#%% print
utils.patch_grid(normalized,width=3,save_name=output_dir + '/Vahadane.pdf')

#%%
utils.show_colors(n.target_stains())
plt.savefig(output_dir + '/stains.pdf')

#%%
utils.show(i1,now=False)
plt.savefig(output_dir + '/i1.pdf')

#%%
print(np.percentile(i1,90))
utils.show(utils.standardize_brightness(i1))
