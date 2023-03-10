# %%
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.io import fits
from bepsf.image import PixelImage
from jax.config import config
import pandas as pd

config.update('jax_enable_x64', True)

#static 12 mag image
static_data = "../../data/public/static_image.12.0.00.fits"
position_data = "../../data/public/spcsv_20_1968_10_20_12.0.csv"

# %%
pos = pd.read_csv(position_data, delimiter=",")
xtrue = pos["x pixel"].values - 1
ytrue = pos["y pixel"].values - 1

# %%
tip = 5
i = 21
xc = xtrue[i]
yc = ytrue[i]

dat = fits.open(static_data)
img = (dat[0].data)
fig = plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.plot(xtrue, ytrue, "*", color="red")
plt.savefig("pixcoord.png")
plt.show()
plt.close()

#import sys
#sys.exit()
# %%
Ny, Nx = np.shape(img)
image_obs = PixelImage(Nx, Ny)
image_obs.Z = np.array(img, dtype=float)
image_obs.Zerr = np.sqrt(image_obs.Z)
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

fap, xap, yap = image_obs.aperture_photometry(xcenters, ycenters,
                                              source_half_extent)

# %%
print("#=",np.sum(image_obs.mask))
# %%

from skimage.segmentation import mark_boundaries

mask_boundary = mark_boundaries(image_obs.Z,
                                image_obs.mask,
                                color=(0, 1, 1),
                                mode='thin')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')
ax.imshow(np.log10(image_obs.Z), cmap="gray")
ax.imshow(mask_boundary, alpha=0.2)
plt.ylim(25, 75)
#plt.xlim(25,75)
plt.savefig("imag.png")
plt.close()
#plt.show()

# %%
#fap, xap, yap = image_obs.aperture_photometry(xcenters, ycenters, source_half_extent)
image_obs.lnfinit = np.log(np.array(fap))
image_obs.xinit = xap
image_obs.yinit = yap
image_obs.idx_anchor = choose_anchor(image_obs,
                                     xcenters,
                                     ycenters,
                                     lnfluxes=None,
                                     plot=True,
                                     mad_threshold=10)

# %%
fap
# %%
from bepsf.utils import check_anchor

check_anchor(image_obs)
plt.ylim(0, 100)
plt.xlim(0, 200)
plt.savefig("anchor.png")
plt.show()
plt.close()

# %%
# Define grid PSF model
from bepsf.psfmodel import GridePSFModel

psf_full_extent = source_half_extent * 2
dx, dy = 1. / 5., 1. / 5.
gridpsf = GridePSFModel(psf_full_extent, psf_full_extent, dx, dy)

# %%
from bepsf.infer import optimize
from bepsf.utils import drop_anchor

res = optimize(gridpsf,
               image_obs,
               xyclim=[-source_half_extent, source_half_extent])
popt, state = res
popt = drop_anchor(popt, image_obs.idx_anchor)
# %%

from bepsf.utils import check_solution

fluxes = fap  #temporarby
check_solution(image_obs,
               xtrue[0:Ntarget],
               ytrue[0:Ntarget],
               fluxes,
               p=popt,
               savefig_position="pos_a.png")

# %%
print(image_obs.idx_anchor)

# %%
mask1d = image_obs.mask1d
epsf1d_pred, _image1d_pred = gridpsf.predict_mean(popt['fluxes'], popt['xcenters'], popt['ycenters'], 
                            jnp.exp(popt['lnlenx']), jnp.exp(popt['lnleny']), jnp.exp(2*popt['lnamp']), jnp.exp(popt['lnmu']),
                            image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d], image_obs.Z1d[~mask1d], image_obs.Zerr1d[~mask1d])
image1d_pred = np.zeros(image_obs.size)
image1d_pred[~mask1d] = _image1d_pred
# %%
from bepsf.utils import compute_epsf

ds = 0.05
x_offset = image_obs.xinit[image_obs.idx_anchor] - xcenters[
    image_obs.idx_anchor]
y_offset = image_obs.yinit[image_obs.idx_anchor] - ycenters[
    image_obs.idx_anchor]
finegrid = GridePSFModel(psf_full_extent, psf_full_extent, dx=ds, dy=ds)
epsf_pred = gridpsf.evaluate_ePSF(finegrid.X, finegrid.Y, x_offset, y_offset,
                                  epsf1d_pred)
"""                                  
true_epsf = compute_epsf(finegrid, truepsffunc,
                         dict(**{
                             "norm": 1.,
                             "xc": 0,
                             "yc": 0
                         }, **truepsfkws))
"""
# %%
print(len(image_obs.xgrid_center)**2)
# %%
import jax.numpy as jnp
# %%
a = jnp.meshgrid(jnp.array([1,2]),jnp.array([3,4,4]))
print(np.shape(a))
b = jnp.meshgrid(image_obs.xgrid_center,image_obs.ygrid_center)
# %%
finegrid = GridePSFModel(psf_full_extent, psf_full_extent, dx=ds, dy=ds)

print(len(finegrid.X))
# %%
print((psf_full_extent/dx//2)*2 + 1)
# %%
