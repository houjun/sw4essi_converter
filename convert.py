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
    
    #print('plot min/max', x_plot_min, x_plot_max, y_plot_min, y_plot_max, z_plot_min, z_plot_max)

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
    drm_file = h5py.File(drm_filename)
    coordinates = drm_file['Coordinates']
    n_coord = int(coordinates.shape[0] / 3)
    drm_x = np.zeros(n_coord)
    drm_y = np.zeros(n_coord)
    drm_z = np.zeros(n_coord)

    # need to know reference point
    
    for i in range(0, n_coord):
        drm_x[i] = coordinates[i*3]
        drm_y[i] = coordinates[i*3+1]
        drm_z[i] = coordinates[i*3+2]

    if verbose:
        print('Done read %d coordinates, first is (%d, %d, %d), last is (%d, %d, %d)' % 
                                      (n_coord, drm_x[0], drm_y[0], drm_z[0], drm_x[-1], drm_y[-1], drm_z[-1]))
    
    drm_file.close()
    return drm_x, drm_y, drm_z, n_coord

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
    
def interpolate(data, x, y, z, order):
    if len(x) != len(y) or len(x) != len(z):
        print('Error with x, y, z input, length not equal (%d, %d, %d)!' % (len(y), len(x), len(z)))
        return
    if order < 0 or order > 5:
        print('Order can only be between 0 and 5! (%d)' % order)
        return    
    output = ndimage.map_coordinates(data, [x, y, z], order=order, mode='constant')
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
                if verbose:
                    print('skipped ', line)
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
    h5file = h5py.File(h5_fname)
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
    
    
def generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only):
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

    nprint = 0
    for i in range(0, nprint):
        if i == 0:
            print('converted essi coordinate:')
        print('(%d, %d, %d)' % (user_essi_x[i], user_essi_y[i], user_essi_z[i]))

    plot_coords(essi_x0, essi_y0, essi_z0, essi_h, essi_nx, essi_ny, essi_nz, user_essi_x, user_essi_y, user_essi_z)

    if plot_only:
        print('Only generate the plots of input nodes')
        exit(0)

    # Convert to array location (spacing is 1)
    user_essi_x_in_array = user_essi_x / essi_h
    user_essi_y_in_array = user_essi_y / essi_h
    user_essi_z_in_array = user_essi_z / essi_h

    #for i in range(0, len(user_essi_x)):
    #    print('(%.2f, %.2f, %.2f)' % (user_essi_x_in_array[i], user_essi_y_in_array[i], user_essi_z_in_array[i]))

    # If ESSI file is too big, split it and read it iteratively
    max_read_size = 16 * 1024 * 1024 * 1024 # 16GB
    essi_total_size = essi_nx * essi_ny * essi_nz * essi_nt * 8
    n_read = int(np.ceil(essi_total_size/max_read_size))

    #print('nread =', n_read)
    my_size = int(essi_nt/n_read)

    output_acc_all = np.zeros((n_coord*3, essi_nt))
    output_dis_all = np.zeros((n_coord*3, essi_nt))

    stime = float(time.perf_counter())

    for my_t in range(0, n_read):
        my_start = my_t * my_size
        if verbose:
            print('Iter %d/%d, start %d, size %d' % (my_t+1, n_read, my_start, my_size))        
        # Read ESSI data, only the previously computed bounding box
        essi_data_vel_0, essi_data_vel_1, essi_data_vel_2 = get_essi_data_btw_step(essi_fname, my_start, my_start+my_size, verbose)

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
        # timestep from my_start to my_start+my_size
        for my_step in range(0, my_size):
                                  
            if gen_acc:
                for coord_i in range(0, 3):
                    if coord_sys[coord_i] == 'x':
                        acc_data = essi_data_acc_0[my_step, :, :, :]
                    elif coord_sys[coord_i] == '-x':
                        acc_data = -essi_data_acc_0[my_step, :, :, :]
                    elif coord_sys[coord_i] == 'y':
                        acc_data = essi_data_acc_1[my_step, :, :, :]
                    elif coord_sys[coord_i] == '-y':
                        acc_data = -essi_data_acc_1[my_step, :, :, :]
                    elif coord_sys[coord_i] == 'z':
                        acc_data = essi_data_acc_2[my_step, :, :, :]
                    elif coord_sys[coord_i] == '-z':
                        acc_data = -essi_data_acc_2[my_step, :, :, :]
                    output_acc_all[coord_i::3, my_start+my_step] = interpolate(acc_data, user_essi_x_in_array, user_essi_y_in_array, user_essi_z_in_array, order)
            
            if gen_dis:
                for coord_i in range(0, 3):
                    if coord_sys[coord_i] == 'x':
                        dis_data = essi_data_dis_0[my_step, :, :, :]
                    elif coord_sys[coord_i] == '-x':
                        dis_data = -essi_data_dis_0[my_step, :, :, :]                    
                    elif coord_sys[coord_i] == 'y':
                        dis_data = essi_data_dis_1[my_step, :, :, :]  
                    elif coord_sys[coord_i] == '-y':
                        dis_data = -essi_data_dis_1[my_step, :, :, :]                      
                    elif coord_sys[coord_i] == 'z':
                        dis_data = essi_data_dis_2[my_step, :, :, :]                         
                    elif coord_sys[coord_i] == '-z':
                        dis_data = -essi_data_dis_2[my_step, :, :, :]    
                
                    output_dis_all[coord_i::3, my_start+my_step] = interpolate(dis_data, user_essi_x_in_array, user_essi_y_in_array, user_essi_z_in_array, order)
  
            if verbose and my_start+my_step % 10000 == 0:
                print('Processed step %d' % (my_start+my_step))
                sys.stdout.flush()
    etime = float(time.perf_counter())

    if verbose:
        print('Done in %.2f seconds.' % (etime-stime))
    return output_acc_all, output_dis_all, essi_timeseq
    
def convert_drm(drm_fname, essi_fname, coord_ref, verbose, plot_only):
    #print('Start time:', datetime.datetime.now().time())
    print('Input  DRM [%s]' %drm_fname)
    print('Input ESSI [%s]' %essi_fname)

    output_fname = drm_fname
    coord_sys = ['y', 'x', '-z']

    gen_dis = True
    gen_acc = True

    user_x, user_y, user_z, n_coord = read_coord_drm(drm_fname, verbose)  
    output_acc_all, output_dis_all, essi_timeseq = generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only)

    print('Ouput DRM [%s]' % output_fname)
    if gen_dis:
        write_to_hdf5(output_fname, '/', 'Accelerations', output_acc_all)
    if gen_acc:
        write_to_hdf5(output_fname, '/', 'Displacements', output_dis_all)
    write_to_hdf5(output_fname, '/', 'Time', essi_timeseq)


def convert_node_tcl(fname, essi_fname, verbose, plot_only):
    #print('Start time:', datetime.datetime.now().time())
    print('Input Node [%s]' %fname)
    print('Input ESSI [%s]' %essi_fname)

    output_fname = fname + '.h5drm'
    
    coord_sys, coord_ref, unit, user_x, user_y, user_z, n_coord = read_input_coord_tcl(fname, verbose)
    
    coordinates = np.zeros(n_coord*3)
    for i in range(0, n_coord):
        coordinates[i] = user_x[i]
        coordinates[i+1] = user_y[i]
        coordinates[i+2] = user_z[i]
    
    write_to_hdf5(output_fname, '/', 'Coordinates', coordinates)
    
    xmin = np.min(user_x)
    xmax = np.max(user_x)
    ymin = np.min(user_y)
    ymax = np.max(user_y)
    zmin = np.min(user_z)
    zmax = np.max(user_z)
    
    is_boundary = np.zeros(n_coord)
    n_boundary = 0
    for i in range(0, n_coord):
        if user_x[i] == xmin or user_x[i] == xmax or user_y[i] == ymin or user_y[i] == ymax or user_z[i] == zmin or user_z[i] == zmax:
            is_boundary[i] = 1
            n_boundary += 1
            
    if verbose:
        print('%d boundary nodes' % n_boundary)
    
    gen_dis = True
    gen_acc = True

    output_acc_all, output_dis_all, essi_timeseq = generate_acc_dis_time(essi_fname, coord_sys, coord_ref, user_x, user_y, user_z, n_coord, gen_acc, gen_dis, verbose, plot_only)

    print('Ouput DRM [%s]' % output_fname)
    if gen_dis:
        write_to_hdf5(output_fname, '/', 'Accelerations', output_acc_all)
    if gen_acc:
        write_to_hdf5(output_fname, '/', 'Displacements', output_dis_all)
    write_to_hdf5(output_fname, '/', 'Time', essi_timeseq)
    
if __name__ == "__main__":
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
    
    #drm_fname = '/global/cscratch1/sd/houhun/essi/3dtest/DRM_input_Model27.hdf5'
    #essi_fname = '/global/cscratch1/sd/houhun/essi/3dtest/3dLocation1Parallel.cycle=00000.essi' 
    if fname == '':
        print('Error, no node coordinate input file is provided, exit...')
        exit(0)        
    if essi_fname == '':
        print('Error, no SW4 ESSI output file is provided, exit...')
        exit(0) 
        
    if use_drm:
        convert_drm(fname, essi_fname, ref_coord, verbose, plotonly)
    elif use_node:
        convert_node_tcl(fname, essi_fname, verbose, plotonly)
