#!/usr/bin/env python3

import sys
import argparse
import h5py
import math
import scipy
from scipy import ndimage
from scipy import integrate
import numpy as np
import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
matplotlib.rcParams['figure.dpi'] = 150
import os
from mpi4py import MPI

# plot a 3D cube and grid points specified by x, y, z arrays
def plot_cube(cube_definition, x, y, z, view):
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

    xs = x + cube_definition[0][1]
    ys = y + cube_definition[0][0]
    zs = z + cube_definition[0][2]
    
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
    fname = 'input_coords' + view + '.png'
    plt.savefig(fname)
    
# Plot user specified grid points along with the ESSI domain, and its relative location in the SW4 domain
def plot_coords(essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, user_essi_x, user_essi_y, user_essi_z):   
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
    plot_cube(cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XYZ')
    plot_cube(cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XZ')
    plot_cube(cube_definition, user_essi_x, user_essi_y, user_essi_z, 'XY')
    
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

    if verbose:
        print('Done read %d coordinates, first is (%d, %d, %d), last is (%d, %d, %d)' % 
                                      (n_coord, drm_x[0], drm_y[0], drm_z[0], drm_x[-1], drm_y[-1], drm_z[-1]))
        print('x, y, z, min/max: (%.0f, %.0f), (%.0f, %.0f), (%.0f, %.0f)' % (np.min(drm_x), np.max(drm_x), np.min(drm_y), np.max(drm_y), np.min(drm_z), np.max(drm_z)) )
    
    drm_file.close()
    return drm_x, drm_y, drm_z, n_coord, internal

def convert_to_essi_coord(coord_sys, from_x, from_y, from_z, ref_essi_xyz, essi_nx, essi_ny, essi_nz):
    from_xyz = [from_x, from_y, from_z]
    for i in range(0, 3):
        if coord_sys[i] == 'x':
            user_essi_x = from_xyz[i] - ref_essi_xyz[0]
        elif coord_sys[i] == '-x':
            user_essi_x = essi_nx - from_xyz[i] + ref_essi_xyz[0]
        elif coord_sys[i] == 'y':
            user_essi_y = from_xyz[i] - ref_essi_xyz[1]
        elif coord_sys[i] == '-y':
            user_essi_y = essi_ny - from_xyz[i] + ref_essi_xyz[1]
        elif coord_sys[i] == 'z':
            user_essi_z = from_xyz[i] - ref_essi_xyz[2]
        elif coord_sys[i] == '-z':
            user_essi_z = essi_nz - from_xyz[i] + ref_essi_xyz[2]
    
    return user_essi_x, user_essi_y, user_essi_z


    
def get_coords_range(x, x_min_val, x_max_val, add_ghost):
    x_min = min(x) - add_ghost
    x_max = max(x) + add_ghost
    if x_min < x_min_val:
        x_min = x_min_val
    if x_max > x_max_val:
        x_max = x_max_val
    
    return x_min, x_max
    
    
def get_essi_meta(essi_fname, verbose):
    # Get parameter values from HDF5 data
    essiout = h5py.File(essi_fname, 'r')
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
    t1 = dt*(nt-1)
    timeseq = np.linspace(t0, t1, nt+1)
    
    if verbose:
        print('ESSI origin x0, y0, z0: ', x0, y0, z0)
        print('grid spacing, h: ', h)
        print('timing, t0, dt, npts, t1: ', t0, round(dt,6), nt, round(t1,6) )
        print('Shape of HDF5 data: ', essiout['vel_0 ijk layout'].shape)
    
    essiout.close()
    
    return x0, y0, z0, h, nx, ny, nz, nt, dt, timeseq

def get_essi_data_btw_step(essi_fname, start, end, verbose):
    stime = float(time.perf_counter())

    essiout = h5py.File(essi_fname, 'r')
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

def get_essi_data_range(essi_fname, xstart, xend, ystart, yend, zstart, zend, verbose):
    stime = float(time.perf_counter())

    essiout = h5py.File(essi_fname, 'r')

    vel_0_all = essiout['vel_0 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    vel_1_all = essiout['vel_1 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    vel_2_all = essiout['vel_2 ijk layout'][:, xstart:xend, ystart:yend, zstart:zend]
    
    essiout.close()
    
    etime = float(time.perf_counter())
    if verbose:
        print('Read from ESSI file took %.2f seconds.' % (etime-stime))
    return vel_0_all, vel_1_all, vel_2_all  

def interpolate(data, x, y, z, order):
    if len(x) != len(y) or len(x) != len(z):
        print('Error with x, y, z input, length not equal (%d, %d, %d)!' % (len(y), len(x), len(z)))
        return
    if order < 0 or order > 5:
        print('Order can only be between 0 and 5! (%d)' % order)
        return    
    
    output = ndimage.map_coordinates(data, [x, y, z], order=order, mode='constant')
    #print(output.shape)
    return output

def read_input_coord_txt(fname, verbose):
    f = open(fname, 'r')
    lines = f.readlines()
    
    max_len = len(lines)
    x = np.zeros(max_len)
    y = np.zeros(max_len)
    z = np.zeros(max_len)
    coord_sys = np.zeros(3)
    coord_ref = np.zeros(3)
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
            coord_ref[0] = float(tmp[0])
            coord_ref[1] = float(tmp[1])
            coord_ref[2] = float(tmp[2])
            if verbose:
                print('Reference Coordinate: (%d, %d, %d)' % (coord_ref[0], coord_ref[1], coord_ref[2]))
            
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
    return coord_sys, coord_ref, unit, x, y, z, n_coord

def read_input_coord_tcl(fname, verbose):
    f = open(fname, 'r')
    lines = f.readlines()
    
    max_len = len(lines)
    x = np.zeros(max_len)
    y = np.zeros(max_len)
    z = np.zeros(max_len)
    coord_sys = np.zeros(3)
    coord_ref = np.zeros(3)
    unit = 'n/a'
    n_coord = 0
    n_skip = 0
    xyzmin = np.zeros(3)
    xyzmax = np.zeros(3)

    i = 0
    # For number of nodes
    while i < max_len:
        line = lines[i]
        if len(line) < 5:
            i+= 1
            continue
            
        if 'Coordinate system' in line:
            coord_sys = lines[i].split(':')[1].split(',')
            for j in range(0, 3):
                coord_sys[j] = coord_sys[j].rstrip()
                coord_sys[j] = coord_sys[j].replace(' ', '')
            if verbose:
                print('Coordinate system: (%s, %s, %s)' % (coord_sys[0], coord_sys[1], coord_sys[2]))
                
        elif 'Valid coordinate range' in line:
            tmp = lines[i].split(':')[1].split(',')
            xyzmin[0] = float(tmp[0])
            xyzmin[1] = float(tmp[1])
            xyzmin[2] = float(tmp[2])
            xyzmax[0] = float(tmp[3])
            xyzmax[1] = float(tmp[4])
            xyzmax[2] = float(tmp[5])
            if verbose:
                print('Valid x, y, z range: (%d, %d, %d) to (%d, %d, %d)' % (xyzmin[0], xyzmin[1], xyzmin[2], xyzmax[0], xyzmax[1], xyzmax[2]))
            
        elif 'Unit' in line:
            unit = lines[i].split(':')[1].strip()
            if verbose:
                print('Unit: (%s)' % unit)
                
        elif 'Reference coordinate' in line:
            tmp = lines[i].split(':')[1].split(',')
            coord_ref[0] = float(tmp[0])
            coord_ref[1] = float(tmp[1])
            coord_ref[2] = float(tmp[2])
            if verbose:
                print('Reference Coordinate offset: (%d, %d, %d)' % (coord_ref[0], coord_ref[1], coord_ref[2]))            
                
        elif line[0:4] == 'node':
            if '#' in line:
                continue
            tmp = line.split()
            x[n_coord] = float(tmp[2])
            y[n_coord] = float(tmp[3])
            z[n_coord] = float(tmp[4])
            # Skip invalid coordinates (based on previously read valid coordinate range)
            if x[n_coord] < xyzmin[0] or x[n_coord] > xyzmax[0] or y[n_coord] < xyzmin[1] or y[n_coord] > xyzmax[1] or z[n_coord] < xyzmin[2] or z[n_coord] > xyzmax[2]:
                n_skip += 1
                i += 1
                #if verbose:
                #    print('skipped ', line)
                continue
            n_coord += 1
            
        i += 1
    if verbose:
        print('Read %d valid nodes, skipped %d nodes' % (n_coord, n_skip))
        print('First (%d, %d, %d), Last (%d, %d, %d)' % (x[0], y[0], z[0], x[n_coord-1], y[n_coord-1], z[n_coord-1]))
    x = np.resize(x, n_coord)
    y = np.resize(y, n_coord)
    z = np.resize(z, n_coord)
    f.close()
    return coord_sys, coord_ref, unit, x, y, z, n_coord

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
    
def create_hdf5(h5_fname, ncoord, nstep, dt):
    h5file = h5py.File(h5_fname, 'w')
    data_grp = h5file.create_group('DRM_Data')
    
    data_location = np.zeros(ncoord, dtype='i4')
    for i in range(0, ncoord):
        data_location[i] = 3*i
    
    dset = data_grp.create_dataset('acceleration', (ncoord*3, nstep), dtype='f4')
    dset = data_grp.create_dataset('data_location', data=data_location, dtype='i4')
    dset = data_grp.create_dataset('displacement', (ncoord*3, nstep), dtype='f4')
    dset = data_grp.create_dataset('internal', (ncoord,), dtype='i4')
    dset = data_grp.create_dataset('xyz', (ncoord, 3), dtype='f4')
    
    data_grp = h5file.create_group('DRM_Metadata')
    dset = data_grp.create_dataset('dt', data=dt, dtype='f8')
    tstart = 0.0
    tend = nstep*dt
    dset = data_grp.create_dataset('tend', data=tend, dtype='f8')
    dset = data_grp.create_dataset('tstart', data=tstart, dtype='f8')
    
    h5file.close()
    
def generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, internal_nodes):
    # Read ESSI metadata
    essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, essi_nt, essi_dt, essi_timeseq = get_essi_meta(essi_fname, verbose)
    essi_x_len_max = (essi_nx-1) * essi_h
    essi_y_len_max = (essi_ny-1) * essi_h
    essi_z_len_max = (essi_nz-1) * essi_h

    #if verbose:
    #    print('ESSI origin x0, y0, z0, h: ', essi_x0, essi_y0, essi_z0, essi_h)
    #    print('ESSI origin nx, ny, nz, nt, dt: ', essi_nx, essi_ny, essi_nz, essi_nt, essi_dt)
    #    print('ESSI max len x, y, z: ', essi_x_len_max, essi_y_len_max, essi_z_len_max)

    # Convert user coordinate to sw4 coordinate, relative to ESSI domain (subset of SW4 domain)
    user_essi_x, user_essi_y, user_essi_z = convert_to_essi_coord(coord_sys, user_x, user_y, user_z, coord_ref, essi_x_len_max, essi_y_len_max, essi_z_len_max)

    # debug print
    nprint = 0
    for i in range(0, nprint):
        if i == 0:
            print('converted essi coordinate:')
        print('(%d, %d, %d)' % (user_essi_x[i], user_essi_y[i], user_essi_z[i]))

    if mpi_rank == 0:
        plot_coords(essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, user_essi_x, user_essi_y, user_essi_z)

    if plot_only:
        if mpi_rank == 0:
            print('Only generate the plots of input nodes')
        exit(0)

    if mpi_rank == 0:
        create_hdf5(output_fname, n_coord, essi_nt, essi_dt)
        
    # Convert to array location (spacing is 1)
    user_essi_x_in_array = user_essi_x / essi_h
    user_essi_y_in_array = user_essi_y / essi_h
    user_essi_z_in_array = user_essi_z / essi_h
    
            
    # Check if we actually need interpolation
    do_interp = False
    for nid in range(0, n_coord):
        if user_essi_x[nid] % essi_h != 0 or user_essi_y[nid] % essi_h != 0 or user_essi_z[nid] % essi_h != 0:
            do_interp = True
            if verbose:
                print('Use spline interpolation.')
            break    

    #for i in range(0, len(user_essi_x)):
    #    print('(%.2f, %.2f, %.2f)' % (user_essi_x_in_array[i], user_essi_y_in_array[i], user_essi_z_in_array[i]))

    
    # Domain decomposition to distribute ESSI data
    if mpi_size <= essi_nx:
        if mpi_size > 1:
            my_x_size = int(essi_nx/(mpi_size-1))
        else:
            my_x_size = int(essi_nx/mpi_size)
        my_x_start = my_x_size * mpi_rank
        my_x_end = my_x_size * mpi_rank + my_x_size
        if my_x_end > essi_nx or mpi_rank == mpi_size-1:
            my_x_end = essi_nx
        my_y_start = 0
        my_y_end = essi_ny
        my_z_start = 0
        my_z_end = essi_nz
    else:
        print('Currently only supports number of MPI ranks no more than ESSI nx (%d)' % essi_nx)
        exit(0)
    if verbose:
        print('Coordinate offset:', coord_ref)
        #print('Rank %d, %d %d, %d %d, %d %d' %(mpi_rank, my_x_start, my_x_end, my_y_start, my_y_end, my_z_start, my_z_end))


    # Find number of nodes in my domain
    my_essi_x_in_array = np.zeros(n_coord)
    my_essi_y_in_array = np.zeros(n_coord)
    my_essi_z_in_array = np.zeros(n_coord)
    my_ncoord = np.zeros(1, dtype='int')
    my_user_coordinates = np.zeros((n_coord,3), dtype='f4')
    is_boundary = np.zeros(n_coord, dtype='i4')

    for nid in range(0, n_coord):
        if user_essi_x_in_array[nid] >= my_x_start and user_essi_x_in_array[nid] < my_x_end and user_essi_y_in_array[nid] >= my_y_start and user_essi_y_in_array[nid] < my_y_end and user_essi_z_in_array[nid] >= my_z_start and user_essi_z_in_array[nid] < my_z_end:
            my_essi_x_in_array[my_ncoord[0]] = user_essi_x_in_array[nid] - my_x_start
            my_essi_y_in_array[my_ncoord[0]] = user_essi_y_in_array[nid] - my_y_start
            my_essi_z_in_array[my_ncoord[0]] = user_essi_z_in_array[nid] - my_z_start
            my_user_coordinates[my_ncoord[0], 0] = user_x[nid]
            my_user_coordinates[my_ncoord[0], 1] = user_y[nid]
            my_user_coordinates[my_ncoord[0], 2] = user_z[nid]
            is_boundary[my_ncoord[0]] = internal_nodes[nid]
            #print('Rank %d: my array idx (%d, %d, %d) <- (%d, %d, %d)' % (mpi_rank, my_essi_x_in_array[my_ncoord], my_essi_y_in_array[my_ncoord], my_essi_z_in_array[my_ncoord], user_x[nid], user_y[nid], user_z[nid]))

            my_ncoord[0] += 1
            
    my_essi_x_in_array.resize(my_ncoord[0])
    my_essi_y_in_array.resize(my_ncoord[0])
    my_essi_z_in_array.resize(my_ncoord[0])
    my_user_coordinates.resize(my_ncoord[0], 3)
    is_boundary.resize(my_ncoord[0])
    
    if verbose:
        print('Rank %d has %d nodes' % (mpi_rank, my_ncoord[0]))
    
    comm = MPI.COMM_WORLD
    all_ncoord = np.empty(mpi_size, dtype='int')
    comm.Allgather([my_ncoord, MPI.INT], [all_ncoord, MPI.INT])
    
    my_offset = 0
    for i in range(0, mpi_rank):
        my_offset += all_ncoord[i]
        
    #if verbose:        
    #    print('Rank %d offset %d ' % (mpi_rank, my_offset))
    
    comm.Barrier()
    
    # Write coordinates and boundary nodes (file is created inside generate_acc_dis_time function)
    # Serialize the write
    if mpi_rank == 0:
        if my_ncoord[0] > 0:
            write_to_hdf5_range_2d(output_fname, 'DRM_Data', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
            write_to_hdf5_range_1d(output_fname, 'DRM_Data', 'internal', is_boundary, my_offset, my_offset+my_ncoord[0])   
        if mpi_size > 1:
            comm.send(my_ncoord, dest=1, tag=11)
    else:
        data = comm.recv(source=mpi_rank-1, tag=11)
        if my_ncoord[0] > 0:
            if verbose:
                print('Rank %d write xyz/internal data' % mpi_rank)
            write_to_hdf5_range_2d(output_fname, 'DRM_Data', 'xyz', my_user_coordinates, my_offset, (my_offset+my_ncoord[0]))
            write_to_hdf5_range_1d(output_fname, 'DRM_Data', 'internal', is_boundary, my_offset, my_offset+my_ncoord[0])    
        if mpi_rank != mpi_size-1:
            comm.send(my_ncoord, dest=mpi_rank+1, tag=11)
            
    comm.Barrier()
    
    output_acc_all = np.zeros((my_ncoord[0]*3, essi_nt), dtype='f4')
    output_dis_all = np.zeros((my_ncoord[0]*3, essi_nt), dtype='f4')    

    if my_ncoord[0] > 0:
        stime = float(time.perf_counter())
        # Read ESSI data, only the previously computed bounding box
        if verbose:
            print('Rank %d start reading data'%mpi_rank)
            sys.stdout.flush()
        essi_data_vel_0, essi_data_vel_1, essi_data_vel_2 = get_essi_data_range(essi_fname, my_x_start, my_x_end, my_y_start, my_y_end, my_z_start, my_z_end, verbose)
        if verbose:
            print('Rank %d finished reading data'%mpi_rank)
            sys.stdout.flush()

        # Convert ESSI data to acceleration and displacement
        if gen_acc:
            essi_data_acc_0 = np.gradient(essi_data_vel_0, essi_dt, axis=0)
            essi_data_acc_1 = np.gradient(essi_data_vel_1, essi_dt, axis=0)
            essi_data_acc_2 = np.gradient(essi_data_vel_2, essi_dt, axis=0)

        if gen_dis:
            essi_data_dis_0 = scipy.integrate.cumtrapz(y=essi_data_vel_0, dx=essi_dt, initial=0, axis=0)
            essi_data_dis_1 = scipy.integrate.cumtrapz(y=essi_data_vel_1, dx=essi_dt, initial=0, axis=0)
            essi_data_dis_2 = scipy.integrate.cumtrapz(y=essi_data_vel_2, dx=essi_dt, initial=0, axis=0)

                

        # Get interpolation result
        order = 1
        if gen_acc:
            for coord_i in range(0, 3):
                if coord_sys[coord_i] == 'x':
                    acc_data = essi_data_acc_0
                elif coord_sys[coord_i] == '-x':
                    acc_data = -essi_data_acc_0
                elif coord_sys[coord_i] == 'y':
                    acc_data = essi_data_acc_1
                elif coord_sys[coord_i] == '-y':
                    acc_data = -essi_data_acc_1
                elif coord_sys[coord_i] == 'z':
                    acc_data = essi_data_acc_2
                elif coord_sys[coord_i] == '-z':
                    acc_data = -essi_data_acc_2

                if do_interp:
                    for ii in range(0, essi_nt):
                        output_acc_all[coord_i::3, ii] = interpolate(acc_data[ii,:,:,:], my_essi_x_in_array, my_essi_y_in_array, my_essi_z_in_array, order)
                        #print('step %d: coord_i=%d, acc[0,0,0]=%f, interp=%f' % (ii, coord_i, acc_data[ii,0,0,0], output_acc_all[0, ii]))
                else:
                    output_acc_all[coord_i::3,:] = acc_data[:, my_essi_x_in_array.astype(int), my_essi_y_in_array.astype(int), my_essi_z_in_array.astype(int)].transpose()


        if gen_dis:
            for coord_i in range(0, 3):
                if coord_sys[coord_i] == 'x':
                    dis_data = essi_data_dis_0
                elif coord_sys[coord_i] == '-x':
                    dis_data = -essi_data_dis_0                   
                elif coord_sys[coord_i] == 'y':
                    dis_data = essi_data_dis_1
                elif coord_sys[coord_i] == '-y':
                    dis_data = -essi_data_dis_1                  
                elif coord_sys[coord_i] == 'z':
                    dis_data = essi_data_dis_2                         
                elif coord_sys[coord_i] == '-z':
                    dis_data = -essi_data_dis_2   
                    
                if do_interp:
                    for ii in range(0, essi_nt):
                        output_dis_all[coord_i::3, ii] = interpolate(dis_data[ii,:,:,:], my_essi_x_in_array, my_essi_y_in_array, my_essi_z_in_array, order)
                else:
                    output_dis_all[coord_i::3,:] = dis_data[:, my_essi_x_in_array.astype(int), my_essi_y_in_array.astype(int), my_essi_z_in_array.astype(int)].transpose()

        etime = float(time.perf_counter())
        if verbose:
            print('Rank %d: finished interpolation in %.2f seconds.' % (mpi_rank, etime-stime)) 
            sys.stdout.flush()
                
    else:
        print('Rank %d has no data to read'%mpi_rank)
    # end if my_ncood[0] > 0
 
    comm.Barrier()

    if mpi_rank == 0:
        if my_ncoord[0] > 0:
            if gen_acc:
                write_to_hdf5_range(output_fname, 'DRM_Data', 'acceleration', output_acc_all, my_offset*3, (my_offset+my_ncoord[0])*3)
            if gen_dis:
                write_to_hdf5_range(output_fname, 'DRM_Data', 'displacement', output_dis_all, my_offset*3, (my_offset+my_ncoord[0])*3)
        if mpi_size > 1:
            comm.send(my_ncoord, dest=1, tag=111)
    else:
        data = comm.recv(source=mpi_rank-1, tag=111)
        if my_ncoord[0] > 0:
            if gen_acc:
                write_to_hdf5_range(output_fname, 'DRM_Data', 'acceleration', output_acc_all, my_offset*3, (my_offset+my_ncoord[0])*3)
            if gen_dis:
                write_to_hdf5_range(output_fname, 'DRM_Data', 'displacement', output_dis_all, my_offset*3, (my_offset+my_ncoord[0])*3)
        if mpi_rank != mpi_size-1:
            comm.send(my_ncoord, dest=mpi_rank+1, tag=111)
            

    return output_acc_all, output_dis_all, essi_timeseq


def convert_node_tcl(fname, essi_fname, plot_only, mpi_rank, mpi_size, verbose):
    if mpi_rank == 0:
        print('Start time:', datetime.datetime.now().time())
        print('Input Node [%s]' %fname)
        print('Input ESSI [%s]' %essi_fname)

    output_fname = fname + '.h5drm'
    
    coord_sys, coord_ref, unit, user_x, user_y, user_z, n_coord = read_input_coord_tcl(fname, verbose)
    
    gen_dis = True
    gen_acc = True
    generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size)

    if mpi_rank == 0:
        print('End time:', datetime.datetime.now().time())

    
def convert_drm(drm_fname, essi_fname, coord_ref, plot_only, mpi_rank, mpi_size, verbose):
    if mpi_rank == 0:
        print('Start time:', datetime.datetime.now().time())
        print('Input  DRM [%s]' %drm_fname)
        print('Input ESSI [%s]' %essi_fname)

    output_fname = drm_fname + '.h5drm'
    coord_sys = ['y', 'x', '-z']

    user_x, user_y, user_z, n_coord, internal_nodes = read_coord_drm(drm_fname, verbose)  
    
    gen_dis = True
    gen_acc = True    
    generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only, output_fname, mpi_rank, mpi_size, internal_nodes)
    
    if mpi_rank == 0:
        print('End time:', datetime.datetime.now().time())
        
if __name__ == "__main__":
    
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    verbose=False
    plotonly=False
    use_drm=False
    use_txt=False
    use_node=False
    essi_fname=''
    fname=''
    ref_coord=np.zeros(3)
    
    parser=argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-d", "--drm", help="full path to the DRM file with node coordinates", default="")
    parser.add_argument("-e", "--essi", help="full path to the SW4 ESSI output file", default="")
    parser.add_argument("-t", "--txt", help="full path to the text file with node coordinates", default="")
    parser.add_argument("-n", "--node", help="full path to the tcl node file", default="")
    parser.add_argument("-p", "--plotonly", help="only generate plots of the input nodes", action="store_true")
    parser.add_argument("-r", "--reference", help="reference node coordinate offset, default 0 0 0", nargs='+', type=float)
    args = parser.parse_args()  
    
    if args.verbose:
        verbose=True
    if args.plotonly:
        plotonly=True
    if args.drm:
        fname=args.drm
        use_drm=True
    if args.txt:
        fname=args.txt
        use_txt=True
    if args.node:
        fname=args.node
        use_node=True
    if args.essi:
        essi_fname=args.essi
    if args.reference:
        ref_coord[0]=args.reference[0]
        ref_coord[1]=args.reference[1]
        ref_coord[2]=args.reference[2]
        
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    
    # only let one rank print info
    if mpi_rank > 0:
        verbose = False
    
    #drm_fname = '/global/cscratch1/sd/houhun/essi/3dtest/DRM_input_Model27.hdf5'
    #essi_fname = '/global/cscratch1/sd/houhun/essi/3dtest/3dLocation1Parallel.cycle=00000.essi' 
    # For 107346node        coord_ref = [24, 24, 24]
    # For 40610node 0722    coord_ref = [-320, -320, 96]
    # For 3dtest            coord_ref = [24, 24, -64] 
    
    if fname == '':
        print('Error, no node coordinate input file is provided, exit...')
        exit(0)        
    if essi_fname == '':
        print('Error, no SW4 ESSI output file is provided, exit...')
        exit(0) 
        
    if use_drm:
        convert_drm(fname, essi_fname, ref_coord, plotonly, mpi_rank, mpi_size, verbose)
    elif use_node:
        convert_node_tcl(fname, essi_fname, plotonly, mpi_rank, mpi_size, verbose)
