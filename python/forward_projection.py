import os, sys, glob
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import torch
import CTorch.utils.geometry as geometry
from CTorch.projector.projector_interface import Projector
from CTorch.reconstructor.fbpreconstructor import FBPReconstructor as FBP
from medpy.io import load
from tqdm import *


def convert_material_to_attn(material, density, energy, total_attn):
    num_materials = total_attn.shape[0] - 1
    ph = np.zeros(material.shape, dtype=np.float32)
    for i in range(num_materials):
        attn = np.interp(energy, total_attn[0, :], total_attn[i + 1, :])
        ph[material == i] = attn

    ph *= density
    ph *= 0.1
    return ph


def spectrum_weights(energy, spectrum, energy1, energy2):
    e1 = np.argmin(np.abs(energy - energy1))
    e2 = np.argmin(np.abs(energy - energy2))
    weights = spectrum[e1:e2]
    weighted_energy = np.sum(energy[e1:e2] * weights) / np.sum(weights)
    return weighted_energy, np.sum(weights)


def spectrum_weights_total(energy, spectrum, energy_division):
    weights = np.zeros(len(energy_division) - 1, dtype=np.float32)
    weighted_energy = np.zeros(len(energy_division) - 1, dtype=np.float32)
    for i in range(len(energy_division) - 1):
        e1 = np.argmin(np.abs(energy - energy_division[i]))
        e2 = np.argmin(np.abs(energy - energy_division[i + 1]))
        weight = spectrum[e1:e2]
        weighted_energy[i] = np.sum(energy[e1:e2] * weight) / np.sum(weight)
        weights[i] = np.sum(weight)
    total_weight = np.sum(weights)
    return weighted_energy, weights, total_weight


def energy_weighted_projection(I0, A, material, density, energy_division, spectrum, total_attn, device='cuda'):
    sino = np.zeros([A.geom.nView, A.geom.nv, A.geom.nu], dtype=np.float32)
    du = A.geom.du
    dv = A.geom.dv
    total_weight = 0
    energy = spectrum[:, 0]
    intensity = spectrum[:, 1]
    weighted_energies, weights, total_weight = spectrum_weights_total(energy, intensity, energy_division)
    for i in range(len(weighted_energies)):
        attn = convert_material_to_attn(material, density, weighted_energies[i], total_attn)
        img = torch.Tensor(attn).reshape([1, 1, nz, ny, nx]).to(device)
        tmp = A(img)
        tmp = tmp.cpu().numpy()[0, 0, :, :, :]
        sino += (I0 * weights[i] / total_weight) * (weighted_energies[i] * 1000.0) * np.exp(-tmp)
    for i in range(A.geom.nView):
        u = np.arange(A.geom.nu) * du - A.geom.nu * du / 2 + A.geom.uOfst.cpu().numpy()[i]
        v = np.arange(A.geom.nv) * dv - A.geom.nv * dv / 2 + A.geom.vOfst.cpu().numpy()[i]
        U, V = np.meshgrid(u, v)
        dist = np.sqrt(A.geom.SDD.cpu().numpy()[i] ** 2 + U ** 2 + V ** 2)
        sino[i, :, :] /= (dist ** 2)
    sino = sino / (du * dv * I0)
    return sino


def recon(method, sino, device='cuda'):
    sino_t = torch.Tensor(sino).reshape([1, 1, sino.shape[0], sino.shape[1]]).to(device)
    imgr = method(sino_t).cpu().numpy()[0, 0, :, :]
    return imgr


def calculate_blank(I0, A, energy_division, spectrum, air_attn, device='cuda'):
    sino = np.zeros([A.geom.nView, A.geom.nv, A.geom.nu], dtype=np.float32)
    du = A.geom.du
    dv = A.geom.dv
    u = np.arange(A.geom.nu) * du - A.geom.nu * du / 2 + A.geom.uOfst.cpu().numpy()[0]
    v = np.arange(A.geom.nv) * dv - A.geom.nv * dv / 2 + A.geom.vOfst.cpu().numpy()[0]
    U, V = np.meshgrid(u, v)

    for i in range(A.geom.nView):
        dist = np.sqrt(A.geom.SDD.cpu().numpy()[i] ** 2 + U ** 2 + V ** 2)
        weighted_energies, weights, total_weight = spectrum_weights_total(
            spectrum[:, 0], spectrum[:, 1], energy_division)
        for j in range(len(weighted_energies)):
            sino[i, :, :] += (I0 * weights[j] / total_weight) * (weighted_energies[j] * 1000) / (dist ** 2)
    sino = sino / (du * dv * I0)

    return sino


def read_spectrum(spectrum_path):
    assert os.path.isfile(spectrum_path), f"{spectrum_path} is not a valid file"
    f = open(spectrum_path, 'r')
    lines = f.readlines()
    f.close()
    energy = np.array([float(l.strip().split()[0]) for l in lines if l[0] != '#'])
    intensity = np.array([float(l.strip().split()[1]) for l in lines if l[0] != '#'])
    energy = energy[:-1]
    intensity = intensity[:-1]
    energy = energy / 1000  # in keV
    intensity /= np.sum(intensity)
    spectrum = np.zeros([len(energy), 2], dtype=np.float32)
    spectrum[:, 0] = energy
    spectrum[:, 1] = intensity
    return spectrum


def read_mcgpu_file(mat_file):
    f = open(mat_file, 'r')
    line = f.readline()
    # print(line)
    while '[MATERIAL NAME]' not in line:
        line = f.readline()
    line = f.readline()
    material_name = line[1:].strip()
    # print(line.strip(), material_name)
    while '[NOMINAL DENSITY' not in line:
        line = f.readline()
    line = f.readline()
    nominal_density = float(line[1:].strip())  # g/cm^3
    # print(line.strip(), nominal_density)
    while '[NUMBER OF DATA VALUES]' not in line:
        line = f.readline()
    line = f.readline()
    num_data_values = int(line[1:].strip())
    # print(line.strip(), num_data_values)
    while '[MEAN FREE PATHS' not in line:
        line = f.readline()
    line = f.readline()
    total_attn = np.zeros([num_data_values, 2], dtype=np.float32)
    for c in range(num_data_values):
        line = f.readline()
        energy = float(line.strip().split()[0]) / 1000  # in keV
        mfp = float(line.strip().split()[4])  # in cm
        mass_attn = (1 / mfp) / nominal_density  # in cm^2/g
        # print(energy, mfp, attn)
        total_attn[c, 0] = energy
        total_attn[c, 1] = mass_attn
    f.close()
    return total_attn


def read_attn(attn_path, material_list):
    assert os.path.isdir(attn_path), f"{attn_path} is not a valid directory"
    for i in range(len(material_list)):
        mat_file = os.path.join(attn_path, material_list[i] + '.mcgpu')
        assert os.path.isfile(mat_file), f"{mat_file} does not exist"
        # print(mat_file)
        attn = read_mcgpu_file(mat_file)
        if i == 0:
            energy = attn[:, 0]
            total_attn = np.zeros([len(material_list) + 1, attn.shape[0]], dtype=np.float32)
            total_attn[0, :] = energy  # air
        total_attn[i + 1, :] = attn[:, 1]

    return total_attn


def read_image(ct_path, mask_path, sad, sdd, detector_size):
    ct, ct_header = load(ct_path)
    print('ct', ct.shape)
    mask, mask_header = load(mask_path)
    ct[mask == 0] = -1000
    mask = np.flip(np.transpose(mask, (2, 1, 0)), axis=0)
    mask = np.flip(mask, axis=1)
    ct = np.flip(np.transpose(ct, (2, 1, 0)), axis=0)
    ct = np.flip(ct, axis=1)
    return ct, mask, ct_header


def convert_image_to_material(image):
    density = HU2Density(image)
    material = Density2Material(density)
    return material, density


def HU2Density(HU):
    HU_range = np.array([0, 8, 24, 520, 904, 1000, 1048, 1128, 1528, 1976,
                        2488, 2824, 3000, 4071, 5000], dtype=np.float32)
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
    material = np.round(np.interp(density, density_range, material_ids)).astype(np.uint8)
    material -= 1
    assert not np.any(np.isnan(material))
    return material


nu = 512  # detector width
nv = 512  # detector height
du = 409.6 / nu   # detecor pixel size (U direction)
dv = 409.6 / nv   # detecor pixel size (V direction)
nView = 720  # number of views
start_angle = 0.0
end_angle = 2 * np.pi
viewAngles = np.arange(start_angle, end_angle, (end_angle - start_angle) / nView)  # projection angle

detType = 'flat'
SAD, SDD = [1000.0], [1536.0]  # source-axis-distance, source-detector-distance
xOfst, yOfst = [0.0], [0.0]  # image center offset
uOfst = [0.0]  # detecor center offset
xSrc, zSrc = [0.0], [0.0]
energy_division = np.arange(10, 100, 2)

projAlgo = 'DD'
reconAlgo = 'ray'

save_path = '../data'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
patient_prefix = '../Patient_images'
mcgpu_material_prefix = '../MCGPU/mcgpu_materials'
material_list = ['air', 'muscle', 'soft_tissue', 'bone', 'cartilage', 'adipose', 'blood',
                 'skin', 'lung', 'gland_others', 'brain', 'red_marrow', 'liver', 'stomach_intestine', 'water']
spectrum_path = '../MCGPU/machines/Spectra/100kVp_XVI.spc'
spectrum = read_spectrum(spectrum_path)
total_attn = read_attn(mcgpu_material_prefix, material_list)
patient_list = sorted(glob.glob(patient_prefix + '/*'))
det_size = np.array([nu * du, nv * dv])
I0 = 1e10
for i in trange(len(patient_list)):
    patient_name = patient_list[i].split('/')[-1]
    print(patient_name)
    if not os.path.isdir(os.path.join(save_path, patient_name)):
        os.makedirs(os.path.join(save_path, patient_name))
    ct_path = os.path.join(patient_list[i], 'ct.mha')
    mask_path = os.path.join(patient_list[i], 'body.mha')
    ct, mask, ct_header = read_image(ct_path, mask_path, SAD[0], SDD[0], det_size)
    dx, dy, dz = ct_header.get_voxel_spacing()
    phantom_origin = np.array(ct.shape) * 0.5 * np.array([dz, dy, dx])
    phantom_origin = np.round(phantom_origin, 3)
    phantom_offset = np.zeros(3, dtype=np.float32)

    z_boundary = int(np.ceil((det_size[1] * SAD[0] / SDD[0]) / dz - 116.0 / dz))
    com = scipy.ndimage.center_of_mass(mask[:z_boundary, :, :]) * np.array([dz, dy, dx])
    com[1] = (ct.shape[1] - 1) * dy - com[1]
    phantom_offset = phantom_origin - com
    xOfst = [phantom_offset[2]]
    yOfst = [-phantom_offset[1]]
    zOfst = [phantom_offset[0]]
    dx = ct_header.get_voxel_spacing()[0]
    dy = ct_header.get_voxel_spacing()[1]
    dz = ct_header.get_voxel_spacing()[2]
    nx = ct.shape[2]
    ny = ct.shape[1]
    nz = ct.shape[0]
    material, density = convert_image_to_material(ct)

    geom = geometry.CircGeom3D(nx, ny, nz, dx, dy, dz, nu, nv, nView, viewAngles, du, dv, detType, SAD, SDD,
                               xOfst=xOfst, yOfst=yOfst, zOfst=zOfst, uOfst=uOfst, xSrc=xSrc, zSrc=zSrc, fixed=True)
    A = Projector(geom, 'proj', projAlgo, 'forward')  # forward projector
    blank = calculate_blank(I0, A, energy_division, spectrum, total_attn[:2, :])
    sino = energy_weighted_projection(I0, A, material, density, energy_division, spectrum, total_attn)
    sino = sino / blank
    sino.tofile(os.path.join(save_path, patient_name, f'proj.raw'))
