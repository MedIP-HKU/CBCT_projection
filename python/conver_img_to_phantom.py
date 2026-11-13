import os, sys, glob
from medpy.io import load
import numpy as np
import scipy
import pydicom
from tqdm import *


def HU2Density(HU):
    HU_range = np.array([0, 8, 24, 520, 904, 1000, 1048, 1128, 1528, 1976, 2488, 2824, 3000, 4071, 5000], dtype=np.float32)
    HU_range -= 1000
    density_range = np.array(
        [0.001, 0.001, 0.001, 0.500, 0.950, 1.000, 1.050, 1.100, 1.334, 1.603, 1.850, 2.100, 4.056, 5.0, 19.300],
        dtype=np.float32)
    density = np.round(np.interp(HU, HU_range, density_range), 3)
    return density


def Density2Material(density):
    density_range = np.array(
        [0, 0.002, 0.00201, 0.8, 0.80001, 0.95, 0.95001, 1.02, 1.02001, 1.04, 1.04001, 1.05, 1.05001, 1.09, 1.09001,
         1.11, 1.11001, 1.2, 1.2001, 5.00], dtype=np.float32)

    material_ids = np.array([1, 1, 9, 9, 6, 6, 3, 3, 11, 11, 13, 13, 2, 2, 5, 5, 12, 12, 4, 4], dtype=np.float32)
    # material = int(np.round(np.interp(density, density_range, material_ids)))
    material = np.round(np.interp(density, density_range, material_ids)).astype(np.uint8)
    assert not np.any(np.isnan(material))
    return material


def read_dcm_images(dcm_list):
    dcm0 = pydicom.dcmread(dcm_list[0])
    nx = dcm0.Rows
    ny = dcm0.Columns
    nz = len(dcm_list)
    dx = dcm0.PixelSpacing[0]
    dy = dcm0.PixelSpacing[1]
    dz = dcm0.SliceThickness
    image = np.zeros([nz, ny, nx], dtype=np.int16)
    for j in range(len(dcm_list)):
        dcm = pydicom.dcmread(dcm_list[j])
        image[dcm.InstanceNumber - 1, :, :] = (dcm.pixel_array * dcm.RescaleSlope) + dcm.RescaleIntercept
    return image, nx, ny, nz, dx, dy, dz


def save_vox(filename, density, material, nx, ny, nz, dx, dy, dz):
    f = open(filename, 'w')
    f.write('# Geometry file created with Create_MCGPU_phantom.m \n# ------- \n[SECTION VOXELS HEADER v.2008-04-13]\n')
    f.write('{:d} {:d} {:d}\n'.format(nx, ny, nz))
    f.write('{:.3f} {:.3f} {:.3f}\n'.format(dx * 0.1, dy * 0.1, dz * 0.1))
    f.write('1\n2\n0\n')
    f.write('[END OF VXH SECTION]\n')
    density_flat = density.flatten()
    material_flat = material.flatten()
    for i in range(len(density_flat)):
        f.write('{:d} {:.3f}\n'.format(material_flat[i], density_flat[i]))
    f.close()


if __name__ == '__main__':
    patient_path = '../MCGPU/images/'
    write_path = '../MCGPU/phantoms/'
    patient_list = sorted(glob.glob(patient_path + '/*'))
    if not os.path.isdir(write_path):
        os.makedirs(write_path)
    for i in trange(len(patient_list), desc='patients'):
        patient_name = patient_list[i].split('/')[-1]
        if not os.path.isdir(write_path + '/' + patient_name):
            os.makedirs(write_path + '/' + patient_name)
        ct, header = load(patient_list[i] + '/CT.nii.gz')
        mask, header2 = load(patient_list[i] + '/body/body.nii.gz')
        ct[mask == 0] = -1000
        ct = np.flip(np.transpose(ct, (2, 1, 0)), axis=0)
        mask = np.flip(np.transpose(mask, (2, 1, 0)), axis=0)
        nx = ct.shape[2]
        ny = ct.shape[1]
        nz = ct.shape[0]
        dx, dy, dz = header.get_voxel_spacing()
        density = HU2Density(ct)
        print(np.max(density))
        material = Density2Material(density)
        outname = write_path + '/' + patient_name + '/ph.vox'
        save_vox(outname, density, material, nx, ny, nz, dx, dy, dz)
