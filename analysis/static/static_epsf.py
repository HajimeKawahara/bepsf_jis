# %%
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.io import fits
from bepsf.image import PixelImage
from jax.config import config
config.update('jax_enable_x64', True)

#static 12 mag image
static_data = "../../data/public/static_image.12.0.00.fits"

# %%
dat = fits.open(static_data)
img = (dat[0].data)
fig = plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()

# %%
Ny, Nx = np.shape(img)
image_obs = PixelImage(Nx, Ny)
image_obs.Z = np.array(img,dtype=float)
Ntarget = 20
Nselect = 1
fac = 0.9955
dx = float(Nx) / float(Ntarget) * fac
dy = float(Ny) / float(Ntarget) * fac

print(Nx, Ny, dx)
#xc = 0.5 * dx + np.array(range(0, Ntarget)) * dx
#yc = 0.5 * dy + np.array(range(0, Nselect)) * dy

xcenters = 0.5 * dx + np.array(range(0, Ntarget)) * dx
ycenters = 0.5 * dy * np.ones_like(xcenters) 
source_half_extent = 10.5


# %%
image_obs.define_mask(xcenters, ycenters, source_half_extent)
# %%
from bepsf.utils import choose_anchor
fap, xap, yap = image_obs.aperture_photometry(xcenters, ycenters, source_half_extent)

# %%

# %%

from skimage.segmentation import mark_boundaries
mask_boundary = mark_boundaries(image_obs.Z,image_obs.mask,color=(0,1,1), mode='thin')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')
ax.imshow(np.log10(image_obs.Z), cmap="gray")
ax.imshow(mask_boundary, alpha=0.2)
plt.ylim(25,75)
#plt.xlim(25,75)
plt.show()

# %%image_obs.lnfinit = np.log(np.array(fap))
image_obs.xinit = xap
image_obs.yinit = yap
image_obs.idx_anchor = choose_anchor(image_obs, xcenters, ycenters, lnfluxes=np.log(fluxes), 
                                     plot=True, mad_threshold=10)
# %%
xx
