#!/usr/bin/env python3

# from genericpath import exists
import os
# import sys
import argparse
import h5py
import math
import scipy
from scipy import integrate
import numpy as np
import pandas as pd
import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpi4py import MPI
import functools
print = functools.partial(print, flush=True) # Don't buffer print
import hdf5plugin # Use this when SW4 output uses ZFP compression, can be installed with "pip install hdf5plugin"

# plot a 3D cube and grid points specified by x, y, z arrays
def plot_cube(save_path, cube_definition, x, y, z, view):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='xkcd:grey')
    faces.set_facecolor((0,0,1,0.05))
    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    #[(1500, 500, 0), (1700, 500, 0), (1500, 650, 0), (1500, 500, 200)]
    x_min = cube_definition[0][0]
    x_max = cube_definition[1][0]
    y_min = cube_definition[0][1]
    y_max = cube_definition[2][1]
    z_min = cube_definition[0][2]
    z_max = cube_definition[3][2]
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min

    x_plot_min = x_min
    x_plot_max = x_max
    y_plot_min = y_min
    y_plot_max = y_max
    z_plot_min = z_min
    z_plot_max = z_max
    
    #print('plot min/max %.2f %.2f %.2f %.2f %.2f %.2f' %( x_plot_min, x_plot_max, y_plot_min, y_plot_max, z_plot_min, z_plot_max))

    x_y_len_diff = abs(x_len-y_len)
    if x_len < y_len:
        x_plot_min = x_min - x_y_len_diff/2
        x_plot_max = x_max + x_y_len_diff/2
    elif x_len > y_len:
        y_plot_min = y_min - x_y_len_diff/2
        y_plot_max = y_max + x_y_len_diff/2
    else:
        tmp0 = 0.95
        tmp1 = 1+(1-tmp0)
        x_plot_min *= tmp0
        x_plot_max *= tmp1
        y_plot_min *= tmp0
        y_plot_max *= tmp1
        
    #print('plot min/max', x_plot_min, x_plot_max, y_plot_min, y_plot_max, z_plot_min, z_plot_max)
       
    ax.set_xlabel('Y(SW4)')
    ax.set_ylabel('X(SW4)')
    ax.set_zlabel('Z(SW4)')
    ax.set_xlim(x_plot_min, x_plot_max) 
    ax.set_ylim(y_plot_min, y_plot_max) 
    ax.set_zlim(z_plot_max, 0) 
    
    lblsize = 5
    ax.zaxis.set_tick_params(labelsize=lblsize)
    ax.yaxis.set_tick_params(labelsize=lblsize)
    ax.xaxis.set_tick_params(labelsize=lblsize)
    
    ax.dist = 12
    #ax.set_aspect('equal')
    
    ax.text(cube_definition[2][0], cube_definition[2][1], cube_definition[2][2]-z_len*.05, 'SW4-ESSI domain', fontsize=7)

    xcolor = 'xkcd:azure'
    ycolor = 'xkcd:green'
    zcolor = 'xkcd:goldenrod'
    xyzmarker = 'x'
    xyzalpha = 0.1
    markersize=2

    # xs = x + cube_definition[0][1]
    # ys = y + cube_definition[0][0]
    # zs = z + cube_definition[0][2]
    xs = x
    ys = y
    zs = z
    
    #print(xs)
    #print(ys)
    #print(zs)
    
    ax.scatter(ys, xs, zs, c='r', marker='.')
    ax.plot(ys, zs, linestyle = 'None', marker=xyzmarker, markersize=markersize, color=ycolor, zdir='y', zs=y_plot_max, alpha=xyzalpha)
    ax.plot(xs, zs, linestyle = 'None', marker=xyzmarker, markersize=markersize,  color=xcolor, zdir='x', zs=x_plot_min, alpha=xyzalpha)
    ax.plot(ys, xs, linestyle = 'None', marker=xyzmarker, markersize=markersize,  color=zcolor, zdir='z', zs=z_plot_max, alpha=xyzalpha)
    
    if view == 'XZ':
        ax.view_init(azim=0, elev=0)    # XZ
    elif view == 'XY':
        ax.view_init(azim=0, elev=90)   # XY
    #ax.view_init(azim=0, elev=-90)   # XZ
    fname = save_path + '/input_coords' + view + '.png'
    plt.savefig(fname)
    
# Plot user specified grid points along with the ESSI domain, and its relative location in the SW4 domain
def plot_coords(essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, user_essi_x, user_essi_y, user_essi_z, save_path='./'):   
    sw4_start_x = essi_x0
    sw4_end_x   = essi_x0 + (essi_nx-1)*essi_h
    sw4_start_y = essi_y0
    sw4_end_y   = essi_y0 + (essi_ny-1)*essi_h
    sw4_start_z = essi_z0
    sw4_end_z   = essi_z0 + (essi_nz-1)*essi_h

    cube_definition = [ (sw4_start_y,sw4_start_x,sw4_start_z), 
                        (sw4_end_y,sw4_start_x,sw4_start_z), 
                        (sw4_start_y,sw4_end_x,sw4_start_z), 
                        (sw4_start_y,sw4_start_x,sw4_end_z)   ]

    # print(cube_definition)
    plot_cube(save_path, cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XYZ')
    plot_cube(save_path, cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XZ')
    plot_cube(save_path, cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XY')
    
def read_coord_drm(drm_filename, verbose):
    if verbose:
        print('Reading coordinates from input file [%s]' % drm_filename)

    # Get the coordinates from DRM file
    drm_file = h5py.File(drm_filename, 'r')
    coordinates = drm_file['Coordinates']
    n_coord = int(coordinates.shape[0] / 3)
    drm_x = np.zeros(n_coord)
    drm_y = np.zeros(n_coord)
    drm_z = np.zeros(n_coord)
    internal = drm_file['Is Boundary Node'][:]

    # need to know reference point
    
    for i in range(0, n_coord):
        drm_x[i] = coordinates[i*3]
        drm_y[i] = coordinates[i*3+1]
        drm_z[i] = coordinates[i*3+2]


    drm_file.close()
    return drm_x, drm_y, drm_z, n_coord, internal

# changed ref coord as just offsets
def convert_to_essi_coord(coord_sys, from_x, from_y, from_z, ref_essi_xyz):
    from_xyz = [from_x, from_y, from_z]
    for i in range(0, 3):
        if coord_sys[i] == 'x':
            # user_essi_x = from_xyz[i] - ref_essi_xyz[0]
            user_essi_x = from_xyz[i] + ref_essi_xyz[0]
        elif coord_sys[i] == '-x':
            # user_essi_x = essi_nx - from_xyz[i] + ref_essi_xyz[0]
            user_essi_x = - from_xyz[i] + ref_essi_xyz[0]
        elif coord_sys[i] == 'y':
            # user_essi_y = from_xyz[i] - ref_essi_xyz[1]
            user_essi_y = from_xyz[i] + ref_essi_xyz[1]
        elif coord_sys[i] == '-y':
            # user_essi_y = essi_ny - from_xyz[i] + ref_essi_xyz[1]
            user_essi_y = - from_xyz[i] + ref_essi_xyz[1]
        elif coord_sys[i] == 'z':
            # user_essi_z = from_xyz[i] - ref_essi_xyz[2]
            user_essi_z = from_xyz[i] + ref_essi_xyz[2]
        elif coord_sys[i] == '-z':
            # user_essi_z = essi_nz - from_xyz[i] + ref_essi_xyz[2]
            user_essi_z = - from_xyz[i] + ref_essi_xyz[2]
    
    return user_essi_x, user_essi_y, user_essi_z

    
def get_coords_range(x, x_min_val, x_max_val, add_ghost):
    x_min = min(x) - add_ghost
    x_max = max(x) + add_ghost
    if x_min < x_min_val:
        x_min = x_min_val
    if x_max > x_max_val:
        x_max = x_max_val
    
    return x_min, x_max


def get_csv_meta(csv_fname):
  # Get parameter values from csv setting file
  df = pd.read_csv(csv_fname)
  # reference point, which is the ESSI or OpenSees origin in the SW4 coordinate system
  ref_coord = np.zeros(3)
  ref_coord[0] = df['essiXstart'][0]
  ref_coord[1] = df['essiYstart'][0]
  ref_coord[2] = df['essiZstart'][0]
  # start time and end time for truncation
  start_t = df['startTime'][0]
  end_t = df['endTime'][0]
  # rotation angle
  rotate_angle = df['rotationAngle'][0]
  zeroMotionDir = df['zeroMotionDir'][0]

  # print('In csv file: ref_coord, start_t, end_t, rotate_angle:', ref_coord, start_t, end_t, rotate_angle)

  return ref_coord, start_t, end_t, rotate_angle, zeroMotionDir


def rotate_coords_ops_xyplane(x, y, z, rotate_angle, ref_coord=[0,0,0]):
  # rotate the coordinates in the OpenSees xy plane around the vertical axis 
  # passing the reference point, rotation positive when counterclockwise
  # Note: (1) rotate the coordinates in a coordinate system is equivalent to 
  #           rotate the coordinate system itself in the opposite direction;
  #       (2) rotate_angle is in degrees;
  #       (3) ref_coord is the coords of one node on the vertical rotate axis, 
  #           default is the OpenSees system origin;

  xyz = np.c_[x-ref_coord[0], y-ref_coord[1], z]
  # print('xyz:', xyz)

  # rotation matrix
  c = np.cos(rotate_angle/180.*np.pi)
  s = np.sin(rotate_angle/180.*np.pi)
  rotationMatrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  # print('rotationMatrix:', rotationMatrix)
  rotated_xyz = np.transpose(np.matmul(rotationMatrix, np.transpose(xyz)))
  # print('rotated_xyz:', rotated_xyz)
  
  return rotated_xyz[:,0], rotated_xyz[:,1], rotated_xyz[:,2]


def get_essi_meta(ssi_fname, verbose):
    # Get parameter values from HDF5 data
    essiout = h5py.File(ssi_fname, 'r')
    h  = essiout['ESSI xyz grid spacing'][0]
    x0 = essiout['ESSI xyz origin'][0]
    y0 = essiout['ESSI xyz origin'][1]
    z0 = essiout['ESSI xyz origin'][2]
    t0 = essiout['time start'][0]
    dt = essiout['timestep'][0]
    nt = essiout['vel_0 ijk layout'].shape[0]
    nx = essiout['vel_0 ijk layout'].shape[1]
    ny = essiout['vel_0 ijk layout'].shape[2]
    nz = essiout['vel_0 ijk layout'].shape[3]
    t1 = t0 + dt*(nt-1)
    timeseq = np.linspace(t0, t1, nt)
    # print('dt, t0, t1, timeseq =', dt, t0, t1, timeseq)
    essiout.close()
    
    return x0, y0, z0, h, nx, ny, nz, nt, dt, timeseq

def get_essi_data_btw_step(ssi_fname, start, end, verbose):
    stime = float(time.perf_counter())

    essiout = h5py.File(ssi_fname, 'r')
    nt = essiout['vel_0 ijk layout'].shape[0]
    if start < 0:
        print('start cannot be negative!', start)
        return
    if end > nt:
        end = nt
    vel_0_all = essiout['vel_0 ijk layout'][start:end, :, :, :]
    vel_1_all = essiout['vel_1 ijk layout'][start:end, :, :, :]
    vel_2_all = essiout['vel_2 ijk layout'][start:end, :, :, :]
    
    essiout.close()
    
    etime = float(time.perf_counter())
    if verbose:
        print('Read from ESSI file took %.2f seconds.' % (etime-stime))
    return vel_0_all, vel_1_all, vel_2_all

def get_essi_data_range(ssi_fname, xstart, xend, ystart, yend, zstart, zend, verbose):
    stime = float(time.perf_counter())

    essiout = h5py.File(ssi_fname, 'r')

    vel_0_all = essiout['vel_0 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    vel_1_all = essiout['vel_1 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    vel_2_all = essiout['vel_2 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    
    essiout.close()
    
    etime = float(time.perf_counter())
    if verbose:
        print('Read from ESSI file took %.2f seconds.' % (etime-stime))
    return vel_0_all, vel_1_all, vel_2_all  

def read_input_coord_txt(fname, verbose):
    f = open(fname, 'r')
    lines = f.readlines()
    
    max_len = len(lines)
    x = np.zeros(max_len)
    y = np.zeros(max_len)
    z = np.zeros(max_len)
    coord_sys = np.zeros(3)
    ref_coord = np.zeros(3)
    unit = 'n/a'
    n_coord = 0

    i = 0
    # For number of nodes
    while i < max_len:
        line = lines[i]
        if 'Coordinate system' in line:
            i += 1
            coord_sys = lines[i].split(',')
            for j in range(0, 3):
                coord_sys[j] = coord_sys[j].rstrip()
                coord_sys[j] = coord_sys[j].replace(' ', '')
            if verbose:
                print('Coordinate system: (%s, %s, %s)' % (coord_sys[0], coord_sys[1], coord_sys[2]))
                
        elif 'Reference coordinate' in line:
            i += 1
            tmp = lines[i].split(',')
            ref_coord[0] = float(tmp[0])
            ref_coord[1] = float(tmp[1])
            ref_coord[2] = float(tmp[2])
            if verbose:
                print('Reference Coordinate: (%d, %d, %d)' % (ref_coord[0], ref_coord[1], ref_coord[2]))
            
        elif 'Unit' in line:
            i += 1
            unit = lines[i].rstrip()
            if verbose:
                print('Unit: (%s)' % unit)
            
        elif 'Coordinates' in line:
            #print('Coordinate:')
            while(i < max_len - 1):
                i += 1
                if '#' in lines[i]:
                    break
                tmp = lines[i].split(',')
                x[n_coord] = float(tmp[0])
                y[n_coord] = float(tmp[1])
                z[n_coord] = float(tmp[2])
                #print('(%d, %d, %d)' % (x[n_coord], y[n_coord], z[n_coord]))
                n_coord += 1
            
        i += 1
    if verbose:
        print('Read %d coordinates' % n_coord)
        print('First (%d, %d, %d), Last (%d, %d, %d)' % (x[0], y[0], z[0], x[n_coord-1], y[n_coord-1], z[n_coord-1]))
    x = np.resize(x, n_coord)
    y = np.resize(y, n_coord)
    z = np.resize(z, n_coord)
    f.close()
    return coord_sys, ref_coord, unit, x, y, z, n_coord


def write_to_hdf5(h5_fname, gname, dname, data):
    h5file = h5py.File(h5_fname, 'r+')
    if gname == '/':
        if dname in h5file.keys():
            dset = h5file[dname]
        else:
            dset = h5file.create_dataset(dname, data.shape, dtype='f4')
    else:
        if gname in h5file.keys():
            grp = h5file[gname]
        else:
            grp = h5file.create_group(gname)
        if dname in grp.keys():
            dset = grp[dname]
        else:
            dset = grp.create_dataset(dname, data.shape, dtype='f4')

    dset[:] = data[:]
    h5file.close()

def write_to_hdf5_range(h5_fname, gname, dname, data, mystart, myend):
    h5file = h5py.File(h5_fname, 'r+')
    if gname == '/':
        dset = h5file[dname]
    else:
        grp = h5file[gname]
        dset = grp[dname]

    #print('write_to_hdf5_range, data shape:', data.shape, 'dset shape:', dset.shape)
    #print('mystart=%d, myend=%d' %(mystart, myend))
    dset[mystart:myend, :] = data[:]
    h5file.close()

def write_to_hdf5_range_1d(h5_fname, gname, dname, data, mystart, myend):
    h5file = h5py.File(h5_fname, 'r+')
    if gname == '/':
        dset = h5file[dname]
    else:
        grp = h5file[gname]
        dset = grp[dname]

    #print('mystart=%d, myend=%d' %(mystart, myend))
    dset[mystart:myend] = data[:]
    h5file.close()
    
def write_to_hdf5_range_2d(h5_fname, gname, dname, data, mystart, myend):
    h5file = h5py.File(h5_fname, 'r+')
    if gname == '/':
        dset = h5file[dname]
    else:
        grp = h5file[gname]
        dset = grp[dname]

    #print('mystart=%d, myend=%d' %(mystart, myend))
    dset[mystart:myend,:] = data[:,:]
    h5file.close()
    
def create_hdf5_opensees(h5_fname, ncoord, nstep, dt, gen_vel, gen_acc, gen_dis, extra_dname):
    h5file = h5py.File(h5_fname, 'w')
    data_grp = h5file.create_group('DRM_Data')
    
    data_location = np.zeros(ncoord, dtype='i4')
    for i in range(0, ncoord):
        data_location[i] = 3*i
    
    if gen_vel:
        dset = data_grp.create_dataset('velocity', (ncoord*3, nstep), dtype='f4')
    if gen_acc:        
        dset = data_grp.create_dataset('acceleration', (ncoord*3, nstep), dtype='f4')
    if gen_dis:        
        dset = data_grp.create_dataset('displacement', (ncoord*3, nstep), dtype='f4')
    dset = data_grp.create_dataset('data_location', data=data_location, dtype='i4')
    dset = data_grp.create_dataset(extra_dname, (ncoord,), dtype='i4')
    dset = data_grp.create_dataset('xyz', (ncoord, 3), dtype='f4')
    
    data_grp = h5file.create_group('DRM_Metadata')
    dset = data_grp.create_dataset('dt', data=dt, dtype='f8')
    tstart = 0.0
    tend = nstep*dt
    dset = data_grp.create_dataset('tend', data=tend, dtype='f8')
    dset = data_grp.create_dataset('tstart', data=tstart, dtype='f8')
    
    h5file.close()

def create_hdf5_csv(h5_fname, ncoord, nstep, dt, gen_vel, gen_acc, gen_dis, extra_dname):
    print('Create HDF5 file with ', ncoord, ' coordinates, ', nstep, ' steps')
    h5file = h5py.File(h5_fname, 'w')

    if gen_vel:
        dset = h5file.create_dataset('velocity', (ncoord*3, nstep), dtype='f4')
    if gen_acc:        
        dset = h5file.create_dataset('acceleration', (ncoord*3, nstep), dtype='f4')
    if gen_dis:        
        dset = h5file.create_dataset('displacement', (ncoord*3, nstep), dtype='f4')

    dset = h5file.create_dataset(extra_dname, (ncoord,), dtype='i4')
    dset = h5file.create_dataset('xyz', (ncoord, 3), dtype='f4')
    
    dset = h5file.create_dataset('dt', data=dt, dtype='f8')
    tstart = 0.0
    tend = nstep*dt
    dset = h5file.create_dataset('tend', data=tend, dtype='f8')
    dset = h5file.create_dataset('tstart', data=tstart, dtype='f8')
    
    h5file.close()

def create_hdf5_essi(h5_fname, ncoord, nstep, dt, gen_vel, gen_acc, gen_dis, extra_dname):
    h5file = h5py.File(h5_fname, 'r+')    
    
    if gen_vel:
        dset = h5file.create_dataset('Velocity', (ncoord*3, nstep), dtype='f4')
    if gen_acc:        
        dset = h5file.create_dataset('Accelerations', (ncoord*3, nstep), dtype='f4')
    if gen_dis:        
        dset = h5file.create_dataset('Displacements', (ncoord*3, nstep), dtype='f4')    
        
    timeseq = np.linspace(0, nstep*dt, nstep+1)    

    h5file.create_dataset('Time', data=timeseq, dtype='i4')
    
    h5file.close()
    
def coord_to_chunkid(x, y, z, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z):
    val = int(np.floor(x/chk_x)*nchk_y*nchk_z + np.floor(y/chk_y)*nchk_z + np.floor(z/chk_z))
    #print('coord_to_chunkid:', x, y, z, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, val)
    return val

def chunkid_to_start(cid, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z):
    #print('cid2:', cid, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
    x = math.floor(cid / (nchk_z * nchk_y))
    y = math.floor((cid - x*nchk_z*nchk_y) / nchk_z)
    z = cid - y*nchk_z - x*nchk_z*nchk_y
    return int(x*chk_x), int(y*chk_y), int(z*chk_z)

def get_chunk_size(ssi_fname):
    fid = h5py.File(ssi_fname, 'r')
    dims = fid['vel_0 ijk layout'].chunks
    if not dims:
        dims = fid['vel_0 ijk layout'].shape
    fid.close()
    #print('Chunk size:', dims)
    return int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
    
def get_nchunk_from_coords(x, y, z, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z):
    if len(x) != len(y) or len(y) != len(z):
        print('Not equal sizes of the x,y,z coordinates array')
    chk_ids = {}
    cnt = 0
    for i in range(0, len(x)):
        cid = coord_to_chunkid(x[i], y[i], z[i], chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
        if cid not in chk_ids:
            chk_ids[cid] = cnt
            cnt += 1
    return len(chk_ids), chk_ids
    
def coord_to_str_3d(x, y, z):
    return str(x)+','+str(y)+','+str(z)

def str_to_coord_3d(s):
    val = s.split(',')
    return int(val[0]), int(val[1]), int(val[2])
    
def allocate_neighbor_coords_8(data_dict, x, y, z, n, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z):
    nadd = 0
    add_cids_dict = {}
    neighbour = 2
    for i0 in range(0, neighbour):
        for i1 in range(0, neighbour):
            for i2 in range(0, neighbour):
                intx, inty, intz = int(x+i0), int(y+i1), int(z+i2)
                coord_str = coord_to_str_3d(intx, inty, intz)
                if coord_str not in data_dict:
                    data_dict[coord_str] = np.zeros(n)
                    nadd += 1

                    cid = coord_to_chunkid(intx, inty, intz, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
                    if cid in add_cids_dict:
                        add_cids_dict[cid].add(coord_str)
                    else:
                        add_cids_dict[cid] = {coord_str}

                    #print(coord_str)
                #else:
                #    print(coord_str, 'alread in dict')
                
    return nadd, add_cids_dict

def read_hdf5_by_chunk(ssi_fname, data_dict, comp, cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose):
    fid = h5py.File(ssi_fname, 'r')
    dset_name = 'vel_' + str(int(comp)) + ' ijk layout'
    for cids_iter in cids_dict:
        # Read chunk
        nread = math.ceil(fid[dset_name].shape[0] / chk_t)
        for start_t in range(0, nread): 
            start_x, start_y, start_z = chunkid_to_start(cids_iter, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
            # print('Read chunk cid =', cids_iter, start_x, chk_x, start_y, chk_y, start_z, chk_z)
            starttime = time.time()
            chk_data = fid[dset_name][int(chk_t*start_t):int(chk_t*(start_t+1)), int(start_x):int(start_x+chk_x), int(start_y):int(start_y+chk_y), int(start_z):int(start_z+chk_z)]
            endtime = time.time()
            if verbose: 
                # print('Rank', mpi_rank, 'read: cid', cids_iter, dset_name, ',time sub chunk', start_t+1, '/', nread, 'time:', endtime-starttime)
                print('Rank {:3d} read: chunk cid {:4d} {}, time slice {:3d}/{} took {:.3f}s'.format(mpi_rank, cids_iter, dset_name, start_t+1, nread, endtime-starttime))
                #sys.stdout.flush()

            starttime = time.time()
            for coord_str in cids_dict[cids_iter]:
                x, y, z = str_to_coord_3d(coord_str)
                # assign values from chunk to data_dict[coord_str][0:3]
                # print('==assign values for', x, y, z, '->', x%chk_x, y%chk_y, z%chk_z, 'cid', cids_iter, 'is in ', cids_iter, 'timestep', chk_t*start_t)
                # print('shape is:', data_dict[coord_str].shape, chk_data.shape)
                data_dict[coord_str][chk_t*start_t:chk_t*(start_t+1)] = chk_data[:,x%chk_x,y%chk_y,z%chk_z]
            endtime = time.time()
            #print('assign value time', endtime-starttime)
    fid.close()
          
def linear_interp(data_dict, x, y, z):
    # see more details at:
    # McCallen et al. Coupling of regional geophysics and local soil-structure 
    # models in the EQSIM fault-to-structure earthquake simulation framework

    # neighbour = 2    
    # neighbour_3d = np.zeros([neighbour,neighbour,neighbour])
    
    xd = x - int(x)
    yd = y - int(y)
    zd = z - int(z)
    # print('x, y, z, xd, yd, zd:', x, y, z, xd, yd, zd)
    if xd > 1 or xd < 0 or yd > 1 or yd < 0 or zd > 1 or zd < 0:
        print('Error with linear interpolation input:', x, y, z)
        exit(0)
        
    c000 = data_dict[coord_to_str_3d(int(x+0), int(y+0), int(z+0))]
    c001 = data_dict[coord_to_str_3d(int(x+0), int(y+0), int(z+1))]
    c010 = data_dict[coord_to_str_3d(int(x+0), int(y+1), int(z+0))]
    c011 = data_dict[coord_to_str_3d(int(x+0), int(y+1), int(z+1))]
    c100 = data_dict[coord_to_str_3d(int(x+1), int(y+0), int(z+0))]
    c101 = data_dict[coord_to_str_3d(int(x+1), int(y+0), int(z+1))]
    c110 = data_dict[coord_to_str_3d(int(x+1), int(y+1), int(z+0))]
    c111 = data_dict[coord_to_str_3d(int(x+1), int(y+1), int(z+1))]
    
    result = ((c000 * (1-xd) + c100 * xd) * (1-yd) + (c010 * (1-xd) + c110 * xd) * yd) * (1-zd) + ((c001 * (1-xd) + c101 * xd) * (1-yd) + (c011 * (1-xd) + c111 * xd) * yd) * zd
    return result

def generate_acc_dis_time(ssi_fname, coord_sys, ref_coord, user_x0, user_y0, user_z0, n_coord, start_t, end_t, rotate_angle, zeroMotionDir, gen_vel, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, extra_data, extra_dname, output_format):
    # Read ESSI metadata
    essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, essi_nt, essi_dt, essi_timeseq = get_essi_meta(ssi_fname, verbose)
    essi_x_len_max = (essi_nx-1) * essi_h
    essi_y_len_max = (essi_ny-1) * essi_h
    essi_z_len_max = (essi_nz-1) * essi_h
    
    # Start and end time step
    if start_t > -1e-6 and end_t > -1e-6:
        start_ts = int(abs(start_t)/essi_dt)
        end_ts   = int(abs(end_t)/essi_dt)
        # if start and end time step equals, we are likely to want all following steps till the end 
        if end_ts > essi_nt or start_ts == end_ts:
            end_ts = int(essi_nt)
        if end_ts <= start_ts:
            print('End time step {} <= start time step {}, no need to extract motions, exit...'.format(end_ts, start_ts))
            exit(0)
    else:
        print('Error getting start and end time step: start_t, end_t, essi_dt =', start_t, end_t, essi_dt)
        exit(0)

    # Save dt, npts for opensees model
    save_path = os.path.dirname(os.path.abspath(output_fname))
    if mpi_rank == 0:
        # os.makedirs(dirname, exist_ok=True)
        # print('save_path=', save_path)
        np.savetxt(save_path + '/Truncated_dt_npts.txt', np.array([[essi_dt, end_ts-start_ts]]), fmt='%.9e %d', header='dt\t\tnpts')

    if verbose and mpi_rank == 0:
        print('\nESSI origin x0, y0, z0, h: ', essi_x0, essi_y0, essi_z0, essi_h)
        print('ESSI origin nx, ny, nz, nt, dt: ', essi_nx, essi_ny, essi_nz, essi_nt, essi_dt)
        print('ESSI max len x, y, z: ', essi_x_len_max, essi_y_len_max, essi_z_len_max)
        print('ESSI max x, y, z: ', essi_x0+essi_x_len_max, essi_y0+essi_y_len_max, essi_z0+essi_z_len_max)
        print('Reference coordinate:', ref_coord)
        print(' ')
        print('Generate output file with timesteps between', start_ts, 'and', end_ts, 'in', output_format, 'format')

    # to get 2D motions for 2D models, modify input node crds to enforce same motion across the width
    # Note: this should be done before rotation, motion zero-out in the out-of-plane direction will be done later
    # TODO: here we use the smallest crd in that direction by default
    user_x, user_y, user_z = user_x0, user_y0, user_z0
    if zeroMotionDir != 'None':
      minCrd = None
      if zeroMotionDir.upper() == 'X':
        minCrd = np.amin(user_x0)
        user_x = np.full(user_x0.shape, minCrd)
      elif zeroMotionDir.upper() == 'Y':
        minCrd = np.amin(user_y0)
        user_y = np.full(user_y0.shape, minCrd)
      elif zeroMotionDir.upper() == 'Z':
        minCrd = np.amin(user_z0)
        user_z = np.full(user_z0.shape, minCrd)

      if mpi_rank == 0:
        print('Zero out motion in {} direction, and for all nodes across that direction, use motion on plane {}={:.4f}'.\
          format(zeroMotionDir, zeroMotionDir, minCrd))

    # Rotate the coordinates in the OpenSees xy plane around the vertical (z) axis
    # rotate/transform only when rotate_angle is other than 0 (default min difference is 1e-2)
    b_rotate = np.where(abs(rotate_angle) > 1e-2, True, False)
    if b_rotate: 
        user_x, user_y, user_z = rotate_coords_ops_xyplane(user_x, user_y, user_z, rotate_angle)

    # Convert user coordinate to sw4 coordinate, relative to ESSI domain (subset of SW4 domain)
    user_essi_x, user_essi_y, user_essi_z = convert_to_essi_coord(coord_sys, user_x, user_y, user_z, ref_coord)

    # debug print
    nprint = 0
    for i in range(0, nprint):
        if i == 0:
            print('converted essi coordinate:')
        print('(%d, %d, %d)' % (user_essi_x[i], user_essi_y[i], user_essi_z[i]))

    # Plot
    if mpi_rank == 0:
        plot_coords(essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, user_essi_x, user_essi_y, user_essi_z, save_path)

    if plot_only:
        if mpi_rank == 0:
            print('Only generate the plots of input nodes')
        exit(0)

    # Check if all node coordinates are within the sw4 domain
    if np.min(user_essi_x) < essi_x0 or np.max(user_essi_x) > essi_x0+essi_x_len_max or \
       np.min(user_essi_y) < essi_y0 or np.max(user_essi_y) > essi_y0+essi_y_len_max or \
       np.min(user_essi_z) < essi_z0 or np.max(user_essi_z) > essi_z0+essi_z_len_max:
        if mpi_rank == 0:
            print('Error: all node coordinates (after rotation) should be within the sw4 domain for extracting the motion')
            print('while:')
            print('\t','Min/Max SW4 x:',essi_x0,essi_x0+essi_x_len_max,'Min/Max user x:',np.min(user_essi_x),np.max(user_essi_x))
            print('\t','Min/Max SW4 y:',essi_y0,essi_y0+essi_y_len_max,'Min/Max user y:',np.min(user_essi_y),np.max(user_essi_y))
            print('\t','Min/Max SW4 z:',essi_z0,essi_z0+essi_z_len_max,'Min/Max user z:',np.min(user_essi_z),np.max(user_essi_z))
            
            debugfile = save_path + '/user_essi_xyz.npy'
            print('\tcheck user_essi_xyz (after rotation) in file \'{}\''.format(debugfile))
            np.save(debugfile, np.c_[user_essi_x, user_essi_y, user_essi_z])
        exit(0)
    
    # if mpi_rank == 0:
    #   print('while user_essi_xyz (after rotation) is:\n', np.c_[user_essi_x, user_essi_y, user_essi_z])
    #   exit(0)

    # Convert to array location (spacing is 1), floating-point
    coord_x = (user_essi_x - essi_x0) / essi_h
    coord_y = (user_essi_y - essi_y0) / essi_h
    coord_z = (user_essi_z - essi_z0) / essi_h  
    
    # Check if we actually need interpolation
    # ghost_cell = 0
    # do_interp = True
    do_interp = False
    for nid in range(0, n_coord):
        if user_essi_x[nid] % essi_h != 0 or user_essi_y[nid] % essi_h != 0 or user_essi_z[nid] % essi_h != 0:
            do_interp = True
            # ghost_cell = 1
            break    
    if mpi_rank == 0:
      if do_interp:
        print('Use spline interpolation.')
      else:
        print('No spline interpolation is needed.')

    # print('Force to not interpolate')
    # do_interp = False

    #for i in range(0, len(user_essi_x)):
    #    print('(%.2f, %.2f, %.2f)' % (coord_x[i], coord_y[i], coord_z[i]))

    chk_t, chk_x, chk_y, chk_z = get_chunk_size(ssi_fname)
    if chk_t <= 0 or chk_x <= 0 or chk_y <= 0 or chk_z <= 0:
        print('Error getting chunk size from essi file', chk_t, chk_x, chk_y, chk_z)
        exit(0)
        
    nchk_x = int(np.ceil(essi_nx/chk_x))
    nchk_y = int(np.ceil(essi_ny/chk_y))
    nchk_z = int(np.ceil(essi_nz/chk_z))
    if nchk_x <= 0 or nchk_y <= 0 or nchk_z <= 0:
        print('Error getting number of chunks', nchk_x, nchk_y, nchk_z)
        exit(0) 
    
    if verbose and mpi_rank == 0:
        print('Essi file: chk_t, chk_x, chk_y, chk_z =', chk_t, chk_x, chk_y, chk_z, ', nchk_x, nchk_y, nchk_z =', nchk_x, nchk_y, nchk_z)

    ntry = 0
    ntry_max = 1
    nchk = 0
    # Try to reduce the chunk size if the number of chunks is less than half the number of ranks
    while(nchk < 0.5*mpi_size):
        if ntry > 0:
            if ntry % 3 == 1 and chk_x % 2 == 0:
                # chk_x /= 2
                chk_x = int(chk_x/2)
            elif ntry % 3 == 2 and chk_y % 2 == 0:
                # chk_y /= 2     
                chk_y = int(chk_y/2)    
            elif ntry % 3 == 0 and chk_z % 2 == 0:
                # chk_z /= 2  
                chk_z = int(chk_z/2)
                
        # Find chunks where all the user input coordinates are (not including adjacent chunks for interpolation yet)
        # cids_dict format: {cid1:index1_in_all_cids,}
        nchk, cids_dict = get_nchunk_from_coords(coord_x, coord_y, coord_z, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
        # if mpi_rank == 0:
        #   print('ntry, nchk, mpi_size, cids_dict, chk_x, chk_y, chk_z = ', ntry, nchk, mpi_size, cids_dict, chk_x, chk_y, chk_z)

        if ntry == 0 and mpi_rank == 0 and nchk != mpi_size:
            print('\nRecommend using', nchk, 'MPI rank(s)', 'instead of currently used', mpi_size, '\n')
        
        # Don't try too manny times
        ntry += 1
        if ntry > ntry_max:
            break

    if verbose and mpi_rank == 0:
        print(nchk, 'total chunks to read/distribute', 'using chunk size (', chk_x, chk_y, chk_z, ')')
        print('All needed chuck ids and their order: cids_dict =', cids_dict)

    # Get the coordinates assigned to this rank
    read_coords_vel_0 = {}
    read_coords_vel_1 = {}
    read_coords_vel_2 = {}
    # coords_str_dict = {}
    is_boundary = np.zeros(n_coord, dtype='i4')
    my_ncoord = np.zeros(1, dtype='int')
    my_user_coordinates = np.zeros((n_coord,3), dtype='f4')
    my_converted_coordinates = np.zeros((n_coord,3), dtype='f4')
    my_cids_dict = {} # format: {cid1:{coord_str1,},}, includes all the chunks for interpolation in this rank

    for i in range(0, n_coord):
        cid = coord_to_chunkid(coord_x[i], coord_y[i], coord_z[i], chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
        if cid < 0:
            print('Error with coord_to_chunkid', coord_x[i], coord_y[i], coord_z[i], cid)
            exit(0)
        # Debug
        if mpi_rank == 0:
            tmp0, tmp1, tmp2 = chunkid_to_start(cid, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
            #print('cid', cid, coord_x[i], coord_y[i], coord_z[i], 'reverse:', tmp0, tmp1, tmp2)
            
        # cids_dict stores the actual unique ids of chunks that contain input coordinates
        if cids_dict[cid] % mpi_size == mpi_rank:
            # if verbose:
            #    print(i, coord_x[i], coord_y[i], coord_z[i], 'goes to chunk', cid, 'and rank', mpi_rank)            
            my_user_coordinates[my_ncoord[0], 0] = user_x0[i]
            my_user_coordinates[my_ncoord[0], 1] = user_y0[i]
            my_user_coordinates[my_ncoord[0], 2] = user_z0[i]
            
            my_converted_coordinates[my_ncoord[0], 0] = coord_x[i]
            my_converted_coordinates[my_ncoord[0], 1] = coord_y[i]
            my_converted_coordinates[my_ncoord[0], 2] = coord_z[i]
            
            coord_str = coord_to_str_3d(int(coord_x[i]), int(coord_y[i]), int(coord_z[i]))
            # coords_str_dict[coord_str] = 1
                    
            # if coord_x[i] % 1 != 0 or coord_y[i] % 1 != 0 or coord_z[i] % 1 != 0:
            if do_interp:
                # Linear interpolation requires 8 neighbours' data
                nadded, add_cids_dict = allocate_neighbor_coords_8(read_coords_vel_0, coord_x[i], coord_y[i], coord_z[i], essi_nt, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
                nadded, add_cids_dict = allocate_neighbor_coords_8(read_coords_vel_1, coord_x[i], coord_y[i], coord_z[i], essi_nt, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)
                nadded, add_cids_dict = allocate_neighbor_coords_8(read_coords_vel_2, coord_x[i], coord_y[i], coord_z[i], essi_nt, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z)

                # print('Rank', mpi_rank, ': add_cids_dict =', add_cids_dict)

                for iadd in add_cids_dict:
                    if iadd in my_cids_dict:
                        my_cids_dict[iadd] |= add_cids_dict[iadd]
                    else:
                        my_cids_dict[iadd] = add_cids_dict[iadd]

                #print(int(coord_x[i]), int(coord_y[i]), int(coord_z[i]), 'added', nadded, 'nodes /', len(read_coords_vel_0))
            else:
                if coord_str not in read_coords_vel_0:
                    read_coords_vel_0[coord_str] = np.zeros(essi_nt)
                    read_coords_vel_1[coord_str] = np.zeros(essi_nt)
                    read_coords_vel_2[coord_str] = np.zeros(essi_nt)

                if cid in my_cids_dict:
                    my_cids_dict[cid].add(coord_str)
                else:
                    my_cids_dict[cid] = {coord_str}
                    
            is_boundary[my_ncoord[0]] = extra_data[i]
            my_ncoord[0] += 1        
        #end if assigned to my rank
    #end for i in all coordinates

    if verbose:
        print('Rank', mpi_rank, 'has my_cids_dict.keys() =', my_cids_dict.keys())

    # Allocated more than needed previously, adjust
    my_user_coordinates.resize(my_ncoord[0], 3)
    my_converted_coordinates.resize(my_ncoord[0], 3)
    is_boundary.resize(my_ncoord[0])
    
    # if mpi_rank == 0:
    #     # print('read_coords_vel_0 =', read_coords_vel_0)
    #     print('Rank', mpi_rank, ': my_converted_coordinates =', my_converted_coordinates)

    comm = MPI.COMM_WORLD
    all_ncoord = np.empty(mpi_size, dtype='int')
    comm.Allgather([my_ncoord, MPI.INT], [all_ncoord, MPI.INT])
    
    my_nchk = len(my_cids_dict)
    if verbose:
        print('Rank', mpi_rank, ': assigned', my_ncoord, 'nodes, need to read', len(read_coords_vel_0), 'nodes, in', my_nchk, 'chunk')

    if my_ncoord[0] > 0:
    # Read data by chunk and assign to read_coords_vel_012
      for dim_iter in range(0, 3):
        if coord_sys[dim_iter] == 'x':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_0, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
        elif coord_sys[dim_iter] == '-x':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_0, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
          for vel_iter in read_coords_vel_0:
            read_coords_vel_0[vel_iter][:] *= -1

        elif coord_sys[dim_iter] == 'y':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_1, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
        elif coord_sys[dim_iter] == '-y':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_1, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
          for vel_iter in read_coords_vel_1:
            read_coords_vel_1[vel_iter][:] *= -1
            
        elif coord_sys[dim_iter] == 'z':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_2, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
        elif coord_sys[dim_iter] == '-z':
          read_hdf5_by_chunk(ssi_fname, read_coords_vel_2, dim_iter, my_cids_dict, chk_x, chk_y, chk_z, nchk_x, nchk_y, nchk_z, chk_t, mpi_rank, verbose)
          for vel_iter in read_coords_vel_2:
            read_coords_vel_2[vel_iter][:] *= -1

      # # debug output
      # if mpi_rank == 0:
      #   import pickle
      #   vel_file0 = open("read_coords_vel_0.pkl", "wb")
      #   pickle.dump(read_coords_vel_0, vel_file0)
      #   vel_file0.close()

      #   vel_file1 = open("read_coords_vel_1.pkl", "wb")
      #   pickle.dump(read_coords_vel_1, vel_file1)
      #   vel_file1.close()

      #   vel_file2 = open("read_coords_vel_2.pkl", "wb")
      #   pickle.dump(read_coords_vel_2, vel_file2)
      #   vel_file2.close()

    # if verbose:
    #     print('Coordinate offset:', ref_coord)
    #     print('Rank %d, %d %d, %d %d, %d %d' %(mpi_rank, my_x_start, my_x_end, my_y_start, my_y_end, my_z_start, my_z_end))

    # Calculate the offset from the global array
    my_offset = 0
    for i in range(0, mpi_rank):
        my_offset += all_ncoord[i]
    # if verbose:        
    #     print('Rank %d offset %d ' % (mpi_rank, my_offset))

    output_acc_all = np.zeros((my_ncoord[0]*3, essi_nt), dtype='f4')
    output_dis_all = np.zeros((my_ncoord[0]*3, essi_nt), dtype='f4')   
    output_vel_all = np.zeros((my_ncoord[0]*3, essi_nt), dtype='f4')    
    
    # Iterate over all coordinates, all the vel data (vel_0 to 2) in read_coords_vel_012 dict for this rank
    if do_interp:
        read_coords_acc_0 = {}
        read_coords_acc_1 = {}
        read_coords_acc_2 = {}

        read_coords_dis_0 = {}
        read_coords_dis_1 = {}
        read_coords_dis_2 = {}
    
        # Convert all data (including 8 neighbours) to acc and dis
        for vel_iter in read_coords_vel_0:
            if gen_acc:
                read_coords_acc_0[vel_iter] =  np.gradient(read_coords_vel_0[vel_iter][:], essi_dt, axis=0)
                read_coords_acc_1[vel_iter] =  np.gradient(read_coords_vel_1[vel_iter][:], essi_dt, axis=0)
                read_coords_acc_2[vel_iter] =  np.gradient(read_coords_vel_2[vel_iter][:], essi_dt, axis=0)
                
            if gen_dis:
                read_coords_dis_0[vel_iter] =  scipy.integrate.cumtrapz(y=read_coords_vel_0[vel_iter][:], dx=essi_dt, initial=0, axis=0)
                read_coords_dis_1[vel_iter] =  scipy.integrate.cumtrapz(y=read_coords_vel_1[vel_iter][:], dx=essi_dt, initial=0, axis=0)
                read_coords_dis_2[vel_iter] =  scipy.integrate.cumtrapz(y=read_coords_vel_2[vel_iter][:], dx=essi_dt, initial=0, axis=0)
                
        # Iterate over all actual coordinates (no neighbour)
        # iter_count = 0
        # for coords_str in coords_str_dict:
        for iter_count in range(0, my_ncoord[0]):
            x = my_converted_coordinates[iter_count, 0]
            y = my_converted_coordinates[iter_count, 1]
            z = my_converted_coordinates[iter_count, 2]
            if gen_acc:
                output_acc_all[iter_count*3+0, :] = linear_interp(read_coords_acc_0, x, y, z)
                output_acc_all[iter_count*3+1, :] = linear_interp(read_coords_acc_1, x, y, z)
                output_acc_all[iter_count*3+2, :] = linear_interp(read_coords_acc_2, x, y, z)
                
            if gen_dis:
                output_dis_all[iter_count*3+0, :] = linear_interp(read_coords_dis_0, x, y, z)
                output_dis_all[iter_count*3+1, :] = linear_interp(read_coords_dis_1, x, y, z)
                output_dis_all[iter_count*3+2, :] = linear_interp(read_coords_dis_2, x, y, z)
                
            if gen_vel:
                output_vel_all[iter_count*3+0, :] = linear_interp(read_coords_vel_0, x, y, z)
                output_vel_all[iter_count*3+1, :] = linear_interp(read_coords_vel_1, x, y, z)
                output_vel_all[iter_count*3+2, :] = linear_interp(read_coords_vel_2, x, y, z)         
                
            # iter_count += 1
    # end if with interpolation                
    else:
        # no interpolation needed, just go through all coordinates' data and convert to acc and dis
        iter_count = 0 
        print('Rank', mpi_rank, 'size of read_coords_vel_0:', len(read_coords_vel_0))
        for iter_count in range(0, my_ncoord[0]):
            x = my_converted_coordinates[iter_count, 0]
            y = my_converted_coordinates[iter_count, 1]
            z = my_converted_coordinates[iter_count, 2]
            coord_str = coord_to_str_3d(int(x), int(y), int(z))

            if gen_acc:
                output_acc_all[iter_count*3+0, :] = np.gradient(read_coords_vel_0[coord_str][:], essi_dt, axis=0)
                output_acc_all[iter_count*3+1, :] = np.gradient(read_coords_vel_1[coord_str][:], essi_dt, axis=0)
                output_acc_all[iter_count*3+2, :] = np.gradient(read_coords_vel_2[coord_str][:], essi_dt, axis=0)
                #if iter_count == 0:
                #    print('acc_0 for', vel_iter, 'is:', output_acc_all[iter_count,:])                
            if gen_dis:
                output_dis_all[iter_count*3+0, :] =  scipy.integrate.cumtrapz(y=read_coords_vel_0[coord_str][:], dx=essi_dt, initial=0, axis=0)
                output_dis_all[iter_count*3+1, :] =  scipy.integrate.cumtrapz(y=read_coords_vel_1[coord_str][:], dx=essi_dt, initial=0, axis=0)
                output_dis_all[iter_count*3+2, :] =  scipy.integrate.cumtrapz(y=read_coords_vel_2[coord_str][:], dx=essi_dt, initial=0, axis=0)   
                #if iter_count == 0:
                #    print('dis_0 for', vel_iter, 'is:', output_dis_all[iter_count,:])                
            if gen_vel:
                output_vel_all[iter_count*3+0, :] = read_coords_vel_0[coord_str][:]
                output_vel_all[iter_count*3+1, :] = read_coords_vel_1[coord_str][:]
                output_vel_all[iter_count*3+2, :] = read_coords_vel_2[coord_str][:]
                # debug
                #if iter_count == 0:
                #    print('vel_0 for', vel_iter, 'is:', read_coords_vel_0[vel_iter])
            # iter_count += 1
        #end for
        print('Rank', mpi_rank, 'has written', iter_count, 'coordinates')
    # end else no interpolation
    
    # transform the motion to be measured in another coordinate system rotated by the specified rotation angle
    if my_ncoord[0] > 0 and b_rotate: 
        c = np.cos(rotate_angle/180.*np.pi)
        s = np.sin(rotate_angle/180.*np.pi)
        transMatrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) # transformation matrix block
        # print('transMatrix = ', transMatrix)
        transMatrixAll = np.zeros((my_ncoord[0]*3, my_ncoord[0]*3), dtype='f4') # for all my_ncoord coordinates
        # print('transMatrixAll.shape=', transMatrixAll.shape)
        for i in range(my_ncoord[0]):
            irange = range(3*i, 3*i+3)
            transMatrixAll[np.ix_(irange, irange)] = transMatrix

        # np.savetxt('transMatrix_mpirank_{}.txt'.format(mpi_rank), transMatrixAll, fmt='%.5e')
        if gen_acc:
            # np.savetxt('acc_mpirank_{}_noT.txt'.format(mpi_rank), output_acc_all[:,0:4], fmt='%.5e')
            np.matmul(transMatrixAll, output_acc_all, output_acc_all)
            # np.savetxt('acc_mpirank_{}_T.txt'.format(mpi_rank), output_acc_all[:,0:4], fmt='%.5e')
        if gen_dis:
            np.matmul(transMatrixAll, output_dis_all, output_dis_all)
        if gen_vel:
            np.matmul(transMatrixAll, output_vel_all, output_vel_all)

    # to get 2D motions for 2D models, zero out motion in the out-of-plane direction
    if zeroMotionDir != 'None':
      zeroMotion1D = np.zeros((1, essi_nt), dtype='f4')
      if zeroMotionDir.upper() == 'X':
        if gen_acc:
          output_acc_all[0::3] = zeroMotion1D
        if gen_dis:
          output_dis_all[0::3] = zeroMotion1D
        if gen_vel:
          output_vel_all[0::3] = zeroMotion1D

      elif zeroMotionDir.upper() == 'Y':
        if gen_acc:
          output_acc_all[1::3] = zeroMotion1D
        if gen_dis:
          output_dis_all[1::3] = zeroMotion1D
        if gen_vel:
          output_vel_all[1::3] = zeroMotion1D

      elif zeroMotionDir.upper() == 'Z':
        if gen_acc:
          output_acc_all[2::3] = zeroMotion1D
        if gen_dis:
          output_dis_all[2::3] = zeroMotion1D
        if gen_vel:
          output_vel_all[2::3] = zeroMotion1D

    # Write coordinates and boundary nodes (file created previously), in serial with baton passing
    comm.Barrier()
    
    if output_format == "OpenSees":
        if mpi_rank == 0:
            create_hdf5_opensees(output_fname, n_coord, end_ts-start_ts, essi_dt, gen_vel, gen_acc, gen_dis, extra_dname)    

            if my_ncoord[0] > 0:
                write_to_hdf5_range_2d(output_fname, 'DRM_Data', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
                write_to_hdf5_range_1d(output_fname, 'DRM_Data', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])   
                if gen_acc:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'acceleration', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'displacement', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
            if mpi_size > 1:
                comm.send(my_ncoord, dest=1, tag=11)
        else:
            data = comm.recv(source=mpi_rank-1, tag=11)
            if my_ncoord[0] > 0:
                if verbose:
                    print('Rank', mpi_rank, 'start to write data')
                write_to_hdf5_range_2d(output_fname, 'DRM_Data', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
                write_to_hdf5_range_1d(output_fname, 'DRM_Data', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])    

                if gen_acc:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'acceleration', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'displacement', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, 'DRM_Data', 'velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
            if mpi_rank != mpi_size-1:
                comm.send(my_ncoord, dest=mpi_rank+1, tag=11) 

    elif output_format == "csv":
        if mpi_rank == 0:
            create_hdf5_csv(output_fname, n_coord, end_ts-start_ts, essi_dt, gen_vel, gen_acc, gen_dis, extra_dname)    

            if my_ncoord[0] > 0:
                write_to_hdf5_range_2d(output_fname, '/', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
                write_to_hdf5_range_1d(output_fname, '/', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])   
                if gen_acc:
                    write_to_hdf5_range(output_fname, '/', 'acceleration', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, '/', 'displacement', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, '/', 'velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
            if mpi_size > 1:
                comm.send(my_ncoord, dest=1, tag=11)
        else:
            data = comm.recv(source=mpi_rank-1, tag=11)
            if my_ncoord[0] > 0:
                if verbose:
                    print('Rank', mpi_rank, 'start to write data')
                write_to_hdf5_range_2d(output_fname, '/', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
                write_to_hdf5_range_1d(output_fname, '/', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])    

                if gen_acc:
                    write_to_hdf5_range(output_fname, '/', 'acceleration', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, '/', 'displacement', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, '/', 'velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
            if mpi_rank != mpi_size-1:
                comm.send(my_ncoord, dest=mpi_rank+1, tag=11) 

    elif output_format == "ESSI":
        if mpi_rank == 0:
            create_hdf5_essi(output_fname, n_coord, end_ts-start_ts, essi_dt, gen_vel, gen_acc, gen_dis, extra_dname)    
            # Write to the template file
            if my_ncoord[0] > 0:
                write_to_hdf5_range_1d(output_fname, '/', 'Coordinates', my_user_coordinates.reshape(my_ncoord[0]*3), my_offset, (my_offset+my_ncoord[0])*3)
                write_to_hdf5_range_1d(output_fname, '/', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])   
                if gen_acc:
                    write_to_hdf5_range(output_fname, '/', 'Accelerations', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, '/', 'Displacements', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, '/', 'Velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)                
                
            if mpi_size > 1:
                comm.send(my_ncoord, dest=1, tag=111)
        else:
            data = comm.recv(source=mpi_rank-1, tag=111)
            if my_ncoord[0] > 0:
                write_to_hdf5_range_1d(output_fname, '/', 'Coordinates', my_user_coordinates.reshape(my_ncoord[0]*3), my_offset, (my_offset+my_ncoord[0])*3)
                write_to_hdf5_range_1d(output_fname, '/', extra_dname, is_boundary, my_offset, my_offset+my_ncoord[0])   
                if gen_acc:
                    write_to_hdf5_range(output_fname, '/', 'Accelerations', output_acc_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_dis:
                    write_to_hdf5_range(output_fname, '/', 'Displacements', output_dis_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)
                if gen_vel:
                    write_to_hdf5_range(output_fname, '/', 'Velocity',     output_vel_all[:,start_ts:end_ts], my_offset*3, (my_offset+my_ncoord[0])*3)              
            if mpi_rank != mpi_size-1:
                comm.send(my_ncoord, dest=mpi_rank+1, tag=111) 
                
    else:
        if mpi_rank == 0:
            print('Invalid output format', output_format)
            
    comm.Barrier()
    if mpi_rank == 0:
        print('Rank', mpi_rank, 'Finished writing data')    
    return
    
def convert_drm(drm_fname, ssi_fname, save_path, ref_coord, start_t, end_t, rotate_angle, zeroMotionDir, plot_only, mpi_rank, mpi_size, verbose):
    if mpi_rank == 0:
        print('Start time:', datetime.datetime.now().time())
        print('Input  DRM [%s]' %drm_fname)
        print('Input ESSI [%s]' %ssi_fname)

    coord_sys = ['y', 'x', '-z']

    # original unrotated node coordinates
    user_x0, user_y0, user_z0, n_coord, extra_data = read_coord_drm(drm_fname, verbose)  
    if verbose and mpi_rank == 0:
        print('Done read %d coordinates, first is (%d, %d, %d), last is (%d, %d, %d)' % (n_coord, user_x0[0], user_y0[0], user_z0[0], user_x0[-1], user_y0[-1], user_z0[-1]))
        print('x, y, z (min/max): (%.0f, %.0f), (%.0f, %.0f), (%.0f, %.0f)' % (np.min(user_x0), np.max(user_x0), np.min(user_y0), np.max(user_y0), np.min(user_z0), np.max(user_z0)) )

    gen_vel = False
    gen_dis = True
    gen_acc = True
    extra_dname = 'internal'
    
    output_format = 'OpenSees'
    output_fname = save_path + '/' + output_format + 'DRMinput.h5drm'

    generate_acc_dis_time(ssi_fname, coord_sys, ref_coord, user_x0, user_y0, user_z0, n_coord, start_t, end_t, rotate_angle, zeroMotionDir,gen_vel, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, extra_data, extra_dname, output_format)
    
    return
        
def convert_csv(csv_fname, ssi_fname, save_path, ref_coord, start_t, end_t, rotate_angle, zeroMotionDir, plot_only, mpi_rank, mpi_size, verbose):
    if mpi_rank == 0:
        print('Start time:', datetime.datetime.now().time())
        print('Input  CSV [%s]' %csv_fname)
        print('Input ESSI [%s]' %ssi_fname)
        
    coord_sys = ['y', 'x', '-z']
    gen_vel = True
    gen_dis = True
    gen_acc = True  
    extra_dname = 'nodeTag'

    # original unrotated node coordinates
    df = pd.read_csv(csv_fname)
    node_tags = df['nodeTag'][:].tolist()
    n_coord = len(node_tags)
    user_x0 = np.zeros(n_coord, dtype='f4')
    user_y0 = np.zeros(n_coord, dtype='f4')
    user_z0 = np.zeros(n_coord, dtype='f4')
    for i in range(0, n_coord):
        user_x0[i] = df.loc[i, 'x']
        user_y0[i] = df.loc[i, 'y']
        user_z0[i] = df.loc[i, 'z']

    if mpi_rank == 0:
        print('Generating motions for %i nodes...' % (n_coord))
    
    output_format = 'csv'
    output_fname = save_path + '/' + output_format + 'NodeMotion.h5drm'

    generate_acc_dis_time(ssi_fname, coord_sys, ref_coord, user_x0, user_y0, user_z0, n_coord, start_t, end_t, rotate_angle, zeroMotionDir,gen_vel, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, node_tags, extra_dname, output_format)
    
    return    

def dframeToDict(dFrame):
    dFrame = list(dFrame.iterrows())
    return {i[1].to_list()[0] : i[1].to_list()[1] for i in dFrame}

def convert_template(csv_fname, template_fname, ssi_fname, start_ts, end_ts, plot_only, mpi_rank, mpi_size, verbose):
    if mpi_rank == 0:
        print('Start time:', datetime.datetime.now().time())
        print('Input  CSV [%s]' %csv_fname)
        print('Input ESSI [%s]' %ssi_fname)
        
    sw4ToESSI_params = dframeToDict(pd.read_csv(csv_fname))
    sw4_i_start = sw4ToESSI_params["sw4_i_start"]
    sw4_i_end   = sw4ToESSI_params["sw4_i_end"]
    sw4_j_start = sw4ToESSI_params["sw4_j_start"]
    sw4_j_end   = sw4ToESSI_params["sw4_j_end"]
    sw4_k_start = sw4ToESSI_params["sw4_k_start"]
    sw4_k_end   = sw4ToESSI_params["sw4_k_end"]    
    essi_x_start = sw4ToESSI_params["essi_x_start"]
    essi_x_end   = sw4ToESSI_params['essi_x_end']
    essi_y_start = sw4ToESSI_params["essi_y_start"]
    essi_y_end   = sw4ToESSI_params["essi_y_end"]
    essi_z_start = sw4ToESSI_params["essi_z_start"]
    essi_z_end   = sw4ToESSI_params["essi_z_end"]
       
    # reference point, which is the ESSI or OpenSees origin in the SW4 coordinate system
    ref_coord[0] = essi_x_start
    ref_coord[1] = essi_y_start
    ref_coord[2] = essi_z_end
        
    coord_sys = ['y', 'x', '-z']
    gen_vel = True
    gen_dis = True
    gen_acc = True 
    extra_dname = 'Is Boundary Node'

    # original unrotated node coordinates
    output_fname = template_fname
    template_file = h5py.File(template_fname)
    coordinates = template_file['Coordinates'][:]
    
    node_tags = template_file['DRM Nodes'][:].tolist()
    n_coord = len(node_tags)
    user_x = np.zeros(n_coord)
    user_y = np.zeros(n_coord)
    user_z = np.zeros(n_coord)
    for i in range(0, n_coord):
        user_x[i] = coordinates[i*3]
        user_y[i] = coordinates[i*3+1]
        user_z[i] = coordinates[i*3+2]
        
    template_file.close()
    
    if verbose and mpi_rank == 0:
        print('Done read %d coordinates, first is (%d, %d, %d), last is (%d, %d, %d)' % (n_coord, user_x[0], user_y[0], user_z[0], user_x[-1], user_y[-1], user_z[-1]))
        print('x, y, z (min/max): (%.0f, %.0f), (%.0f, %.0f), (%.0f, %.0f)' % (np.min(user_x), np.max(user_x), np.min(user_y), np.max(user_y), np.min(user_z), np.max(user_z)) )
        print('Start/end timestep', start_ts, end_ts)
        
    output_format = 'ESSI'
        
    generate_acc_dis_time(ssi_fname, coord_sys, ref_coord, user_x, user_y, user_z, n_coord, start_ts, end_ts, gen_vel, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, node_tags, extra_dname, output_format)
    return    

if __name__ == "__main__":
    
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['PYTHONUNBUFFERED'] = 'TRUE'
    verbose=False
    plotonly=False
    use_drm=False
    use_csv=False
    use_template=False
    ssi_fname=''
    drm_fname=''
    csv_fname=''
    template_fname=''
    save_path='./'
    ref_coord=np.zeros(3)
    start_t=0
    end_t=0
    rotate_angle = 0
    zeroMotionDir = 'None'
    
    parser=argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", help="full path to the CSV setting file", default="")
    parser.add_argument("-d", "--drm", help="full path to the DRM file with node coordinates", default="")
    parser.add_argument("-e", "--essi", help="full path to the SW4 ESSI output file", default="")
    parser.add_argument("-t", "--template", help="full path to the ESSI template file with node coordinates", default="")
    parser.add_argument("-p", "--plotonly", help="only generate plots of the input nodes", action="store_true")
    parser.add_argument("-r", "--reference", help="reference node coordinate offset, default 0 0 0", nargs='+', type=float)
    # parser.add_argument("-s", "--steprange", help="timestep range, default 0 total_steps", nargs='+', type=int)
    parser.add_argument("-s", "--timerange", help="time range, will return all steps after the lower limit for equal upper and lower limit", nargs='+', type=int)
    parser.add_argument("-R", "--rotateanlge", help="rotate angle for node coordinate and motion: [0, 360)", type=float)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-P", "--savepath", help="full path for saving the result files", default="")
    parser.add_argument("-z", "--zeroMotionDir", help="direction for zeroing out motion and enforce same motion across nodes in that direction: None(default), x, y, z", default="")
    args = parser.parse_args()  
    
    if args.verbose:
        verbose=True
    if args.plotonly:
        plotonly=True
    if args.drm:
        drm_fname=args.drm
        use_drm=True
    if args.csv:
        csv_fname=args.csv
        use_csv=True
        # read parameters from the csv setting file
        # Note: As the drm hdf5 file and csv file are usually paired in our structural
        #       analysis workflow, its useful to read the settings from the csv file by default.
        #       These parameters can be OVERWRITTEN when specified in command line if needed.
        ref_coord, start_t, end_t, rotate_angle, zeroMotionDir = get_csv_meta(csv_fname)
    if args.template:
        template_fname=args.template
        use_template=True
    if args.essi:
        ssi_fname=args.essi
    if args.reference:
        ref_coord[0]=args.reference[0]
        ref_coord[1]=args.reference[1]
        ref_coord[2]=args.reference[2]
    if args.timerange:
        start_t=args.timerange[0]
        end_t=args.timerange[1]
    if args.rotateanlge:
        rotate_angle = args.rotateanlge
    if args.savepath:
        save_path = args.savepath
    if args.zeroMotionDir:
        zeroMotionDir = args.zeroMotionDir

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    if mpi_rank == 0:
        print('Running with ', mpi_size, 'MPI processes')
        os.makedirs(save_path, exist_ok=True)
        
    if drm_fname == '' and csv_fname == '' and template_fname == '':
        print('Error, no node coordinate input file is provided, exit...')
        exit(0)
    if ssi_fname == '':
        print('Error, no SW4 ESSI output file is provided, exit...')
        exit(0) 

    if verbose and mpi_rank == 0:
      print('Using ref_coord={}, start_t={}, end_t={}, rotate_angle={} to extract motions'.format(ref_coord, start_t, end_t, rotate_angle))

    if use_drm:
        convert_drm(drm_fname, ssi_fname, save_path, ref_coord, start_t, end_t, rotate_angle, zeroMotionDir, plotonly, mpi_rank, mpi_size, verbose)
    elif use_csv and not use_template:
        convert_csv(csv_fname, ssi_fname, save_path, ref_coord, start_t, end_t, rotate_angle, zeroMotionDir, plotonly, mpi_rank, mpi_size, verbose)
    elif use_csv and use_template:
        convert_template(csv_fname, template_fname, ssi_fname, start_t, end_t, plotonly, mpi_rank, mpi_size, verbose)
        
    if mpi_rank == 0:
        print('End time:', datetime.datetime.now().time())