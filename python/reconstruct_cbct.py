import os, glob
import itk
import numpy as np
from itk import RTK as rtk
from tqdm import *
import matplotlib.pyplot as plt
from medpy.io import load
# Reference: https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction/FirstCudaReconstruction.py


def reconstruct_CBCT(projs, mu_w, sid, sdd, origin, spacing, size, offset_u=0.0):
    CPUImageType = itk.Image[itk.F, 3]
    GPUImageType = itk.CudaImage[itk.F, 3]
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    angular_arc = 360.0
    n_proj = projs.GetLargestPossibleRegion().GetSize()[2]
    firstAngle = 270
    for i in range(itk.size(projs)[2]):
        gantry_angle = firstAngle + i * (angular_arc / n_proj)
        geometry.AddProjection(sid, sdd, gantry_angle, offset_u)

    constantImageSource = rtk.ConstantImageSource[GPUImageType].New()
    constantImageSource.SetOrigin(origin)
    constantImageSource.SetSpacing(spacing)
    constantImageSource.SetSize(size)
    constantImageSource.SetConstant(0.0)
    constantImageSource.Update()

    print("Performing reconstruction")
    projections = GPUImageType.New()
    projections.SetPixelContainer(projs.GetPixelContainer())
    projections.CopyInformation(projs)
    projections.SetBufferedRegion(projs.GetBufferedRegion())
    projections.SetRequestedRegion(projs.GetRequestedRegion())

    ddf = rtk.CudaDisplacedDetectorImageFilter.New()
    ddf.SetInput(projections)
    ddf.SetGeometry(geometry)
    ddf.InPlaceOff()
    ddf.Update()

    feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
    feldkamp.SetGeometry(geometry)
    feldkamp.GetRampFilter().SetTruncationCorrection(0.1)
    feldkamp.GetRampFilter().SetHammingFrequency(5.0)
    feldkamp.SetInput(0, constantImageSource.GetOutput())
    feldkamp.SetInput(1, ddf.GetOutput())
    feldkamp.Update()

    print("Masking field-of-view")
    fov_filter = rtk.FieldOfViewImageFilter[CPUImageType, CPUImageType].New()
    fov_filter.SetOutsideValue(0)
    fov_filter.SetGeometry(geometry)
    fov_filter.SetProjectionsStack(projs)
    fov_filter.SetInput(feldkamp.GetOutput())
    fov_filter.SetDisplacedDetector(True)
    fov_filter.Update()

    print("Permute and flip axis")
    permute_filter = itk.PermuteAxesImageFilter[CPUImageType].New()
    permute_filter.SetInput(fov_filter.GetOutput())
    permute_filter.SetOrder((2, 0, 1))
    permute_filter.Update()

    # print("Applying Median filter")
    # median_filter = itk.MedianImageFilter[CPUImageType, CPUImageType].New()
    # median_filter.SetInput(permute_filter.GetOutput())
    # median_filter.SetRadius([0, 0, 1])
    # median_filter.Update()

    print("Converting into HU")
    sub_filter = itk.SubtractImageFilter[CPUImageType, CPUImageType, CPUImageType].New()
    sub_filter.SetInput(permute_filter.GetOutput())
    sub_filter.SetConstant(mu_w)
    sub_filter.Update()

    div_filter = itk.DivideImageFilter[CPUImageType, CPUImageType, CPUImageType].New()
    div_filter.SetInput(sub_filter.GetOutput())
    div_filter.SetConstant(mu_w / 1000)
    div_filter.Update()

    tresh_abv = itk.ThresholdImageFilter[CPUImageType].New()
    tresh_abv.SetInput(div_filter.GetOutput())
    tresh_abv.ThresholdAbove(3071)
    tresh_abv.SetOutsideValue(3071)
    tresh_abv.Update()
    image = tresh_abv.GetOutput()

    tresh_blw = itk.ThresholdImageFilter[CPUImageType].New()
    tresh_blw.SetInput(tresh_abv.GetOutput())
    tresh_blw.ThresholdBelow(-1000)
    tresh_blw.SetOutsideValue(-1000)
    tresh_blw.Update()
    image = tresh_blw.GetOutput()

    image.SetDirection(constantImageSource.GetDirection())
    image.SetOrigin([origin[0], origin[2], origin[1]])
    print("finished!")
    return image


def read_data(path):
    data = np.load(path)
    primary_rescaler = data['primary_rescaler']
    # primary_mc_rescaler = data['primary_mc_rescaler']
    scatter_rescaler = data['scatter_rescaler']
    primary_only = data['primary'].astype(np.float32) * primary_rescaler[0] + primary_rescaler[1]
    scatter_only = data['scatter'].astype(np.float32) * scatter_rescaler[0] + scatter_rescaler[1]
    return primary_only, scatter_only


path = '../../data/truebeam/brain/HN_P001/'

plot_path = './plots'

sid = 1000.0
sdd = 1500.0
nu = 512
nv = 384
du = 400.0 / nu
dv = 300.0 / nv
offset_u = 0.0
mu_water = 0.02
origin = [-512 * 1.0 * 0.5, -256 * 1.0 * 0.5, -512 * 1.0 * 0.5]
spacing = [1.0, 1.0, 1.0]
size = [512, 256, 512]

if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

patient_name = os.path.basename(path)
proj_list = sorted(glob.glob(os.path.join(path, '*.npz')))
proj_sf = np.zeros([len(proj_list), nv, nu], dtype=np.float32)
proj = np.zeros([len(proj_list), nv, nu], dtype=np.float32)
for i in range(len(proj_list)):
    primary_only, scatter_only = read_data(proj_list[i])
    proj_sf[i, :, :] = -np.log(primary_only)
    proj[i, :, :] = -np.log(primary_only + scatter_only)
proj_itk = itk.image_from_array(proj.astype(np.float32))
proj_itk.SetOrigin([-0.5 * du * nu, -0.5 * dv * nv, 0])
proj_itk.SetSpacing([du, dv, 1])
image = reconstruct_CBCT(proj_itk, mu_water, sid, sdd, origin, spacing, size, offset_u)  # mu_water based on simulations

proj_sf_itk = itk.image_from_array(proj_sf.astype(np.float32))
proj_sf_itk.SetOrigin([-0.5 * du * nu, -0.5 * dv * nv, 0])
proj_sf_itk.SetSpacing([du, dv, 1])
image_sf = reconstruct_CBCT(proj_sf_itk, mu_water, sid, sdd, origin, spacing, size, offset_u)  # mu_water based on simulations

recon_np = itk.array_from_image(image)
recon_sf_np = itk.array_from_image(image_sf)
plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.imshow(recon_np[:, recon_np.shape[1] // 2, :], cmap='gray', vmin=-500, vmax=500)
plt.title('Reconstruction with scatter')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(recon_sf_np[:, recon_sf_np.shape[1] // 2, :], cmap='gray', vmin=-500, vmax=500)
plt.title('Reconstruction without scatter')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.plot(recon_np[recon_np.shape[0] // 2, recon_np.shape[1] // 2, :], 'r-', label='with scatter')
plt.plot(recon_sf_np[recon_sf_np.shape[0] // 2, recon_sf_np.shape[1] // 2, :], 'k--', label='without scatter')
plt.legend()
plt.title('Central profile')

plt.subplot(2, 3, 4)
plt.imshow(recon_np[recon_np.shape[0] // 2, :, :], cmap='gray', vmin=-500, vmax=500)
plt.title('Reconstruction with scatter')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(recon_sf_np[recon_sf_np.shape[0] // 2, :, :], cmap='gray', vmin=-500, vmax=500)
plt.title('Reconstruction without scatter')
plt.axis('off')

plt.tight_layout(pad=1.1)
plt.savefig(os.path.join(plot_path, 'recon_comparison_lung.png'))
