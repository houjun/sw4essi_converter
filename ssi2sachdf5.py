#!/usr/bin/env python

import numpy as np
import h5py
import hdf5plugin
import math
import time
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import scipy
from pyproj import CRS
from pyproj import Transformer
from datetime import datetime
from scipy import integrate
from mpi4py import MPI

#Get parameter values from HDF5 data
def get_essi_meta(ssi_fname):
    ssifile = h5py.File(ssi_fname, 'r')
    h = ssifile['ESSI xyz grid spacing'][0]
    x0 = ssifile['ESSI xyz origin'][0]
    y0 = ssifile['ESSI xyz origin'][1]
    z0 = ssifile['ESSI xyz origin'][2]
    lon0 = ssifile['Grid lon-lat origin'][0]
    lat0 = ssifile['Grid lon-lat origin'][1]
    t0 = ssifile['time start'][0]
    dt = ssifile['timestep'][0]
    az = ssifile['Grid azimuth'][0]
    nt = ssifile['vel_0 ijk layout'].shape[0]
    nx = ssifile['vel_0 ijk layout'].shape[1]
    ny = ssifile['vel_0 ijk layout'].shape[2]
    nz = ssifile['vel_0 ijk layout'].shape[3]
    chk_size = ssifile['vel_0 ijk layout'].chunks

    #Adjustment for downsampling not recored correctly
    if '-100' in ssi_fname:
        dt *= 100
        print("Adjust dt to", dt, flush=True)

    ssifile.close()

    return x0, y0, z0, lon0, lat0, h, nx, ny, nz, nt, az, dt, chk_size


if __name__ == "__main__":
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['PYTHONUNBUFFERED'] = 'TRUE'
    spacing = int(1000)
    verbose = False
    ssi_fname = '/global/cscratch1/sd/houhun/stripe_large/surface-800-100.ssi'
    out_fname = ssi_fname + '.h' + str(spacing) + '.h5'
    out_dir = './'
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    spacing = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="full path to the ssi file",
                        default="")
    parser.add_argument("-o",
                        "--output",
                        help="full path to the output file",
                        default="")
    parser.add_argument("-d",
                        "--directory",
                        help="full path to the image and csv output directory",
                        default="")
    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-x",
                        "--xlimit",
                        help="x lower and upper bound, default 0 0",
                        nargs='+',
                        type=int)
    parser.add_argument("-y",
                        "--ylimit",
                        help="y lower and upper bound, default 0 0",
                        nargs='+',
                        type=int)
    parser.add_argument("-s",
                        "--spacing",
                        help="distance between stations, default 1000",
                        nargs='+',
                        type=int)
    args = parser.parse_args()

    if args.verbose:
        verbose = True
    if args.input:
        ssi_fname = args.input
    if args.output:
        out_fname = args.output
    if args.directory:
        out_dir = args.directory
    if args.spacing:
        spacing = args.spacing[0]
    if args.xlimit:
        xmin = args.xlimit[0]
        xmax = args.xlimit[1]
    if args.ylimit:
        ymin = args.ylimit[0]
        ymax = args.ylimit[1]

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    regen_image = True
    save_image = True
    # Reverse the direction of the velocity values due to the SW4 direction
    velocity_sign = [-1, -1, -1]

    rechdf5_image_dir_name = '/rechdf5_image/'
    rechdf5_csr_dir_name = '/rechdf5_csv/'
    image_dir = out_dir + rechdf5_image_dir_name
    csv_dir = out_dir + rechdf5_csr_dir_name
    dset_names = ['X', 'Y', 'Z']

    vel_min = [-1.2, -1.2, -1.2]
    vel_max = [1.2, 1.2, 1.2]
    dis_min = [-0.5, -0.5, -0.5]
    dis_max = [0.5, 0.5, 0.5]
    acc_min = [-15, -15, -15]
    acc_max = [15, 15, 15]

    fig_width = 7
    fig_height = 3
    fig_dpi = 50

    h = 0
    x0 = 0
    y0 = 0
    z0 = 0
    lon0 = 0
    lat0 = 0
    t0 = 0
    delta = 0
    az = 0
    npts = 0
    nx = 0
    ny = 0
    nz = 0
    chk_size = np.zeros(4)
    origin_lon = 0
    origin_lat = 0

    if mpi_rank == 0:
        print('Running with ', mpi_size, 'MPI processes', flush=True)
        #get ssi metadata
        x0, y0, z0, origin_lon, origin_lat, h, nx, ny, nz, npts, az, delta, chk_size = get_essi_meta(ssi_fname)

    x0 = comm.bcast(x0, root=0)
    y0 = comm.bcast(y0, root=0)
    z0 = comm.bcast(z0, root=0)
    origin_lon = comm.bcast(origin_lon, root=0)
    origin_lat = comm.bcast(origin_lat, root=0)
    h = comm.bcast(h, root=0)
    nx = comm.bcast(nx, root=0)
    ny = comm.bcast(ny, root=0)
    nz = comm.bcast(nz, root=0)
    npts = comm.bcast(npts, root=0)
    az = comm.bcast(az, root=0)
    delta = comm.bcast(delta, root=0)
    chk_size = comm.bcast(chk_size, root=0)

    xmin = int(xmin - x0)
    xmax = int(xmax - x0)
    ymin = int(ymin - y0)
    ymax = int(ymax - y0)

    if verbose and mpi_rank == mpi_size - 1:
        print('h', h)
        print('az', az)
        print('origin_lon', origin_lon)
        print('origin_lat', origin_lat)
        print('chk_size', chk_size)
        print('delta, npts, t1:', delta, npts, t0 + delta * (npts - 1))
        print('origin', x0, y0, z0)
        print('extract domain:', xmin, xmax, ymin, ymax, spacing, flush=True)

    sw4_proj = '+proj=tmerc +datum=NAD83 +lon_0=-123.0 +lat_0=35.0 +scale=0.9996'
    lonlat_proj = '+proj=latlong +datum=NAD83'
    crs_projection = CRS.from_proj4(sw4_proj)
    crs_lonlat = CRS.from_proj4(lonlat_proj)
    transformer2xy = Transformer.from_crs(crs_lonlat, crs_projection)
    transformer2lonlat = Transformer.from_crs(crs_projection, crs_lonlat)
    #print(transformer2lonlat)

    #origin_xy = [71783.71046330612, 375122.4802476847]
    PI = 3.14159265358979323846
    az = az * PI / 180.0
    origin_off_xy = transformer2xy.transform(origin_lon, origin_lat)
    #print('transform to ', origin_off_xy)

    last_sta_y = ymin + spacing
    #Create output file and dsets
    if mpi_rank == 0:
        outfile = h5py.File(out_fname, 'a')

        if not 'DOWNSAMPLE' in outfile.keys():
            downsample = 1
            dset = outfile.create_dataset('DOWNSAMPLE', (1, ), dtype='i4')
            dset[0] = downsample

        if not 'UNIT' in outfile.keys():
            outfile.attrs["UNIT"] = np.string_("m/s")

        if not 'DATETIME' in outfile.keys():
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%dT%H:%M:%S.0")
            outfile.attrs["DATETIME"] = np.string_(dt_string)

        if not 'DELTA' in outfile.keys():
            dset = outfile.create_dataset('DELTA', (1, ), dtype='f4')
            dset[0] = delta

        xyz = np.array([0.0, 0.0, 0.0])
        stlalodp = np.array([0.0, 0.0, 0.0])

        # Check for last station exist, if so, skip creation
        sta_name = 'S_' + str(int(xmax-spacing)) + '_' + str(int(ymax-spacing))
        # if not sta_name in outfile.keys():
        for sta_x in range(xmin, xmax, spacing):
            for sta_y in range(ymin, ymax, spacing):
                sta_name = 'S_' + str(sta_x) + '_' + str(sta_y)

                if sta_name in outfile.keys():
                    grp = outfile[sta_name]
                else:
                    grp = outfile.create_group(sta_name)

                if not 'ISNSEW' in grp.keys():
                    nsew = 0
                    dset = grp.create_dataset('ISNSEW', (1, ), dtype='i4')
                    dset[0] = nsew

                if not 'STX,STY,STZ' in grp.keys():
                    xyz[0] = sta_x
                    xyz[1] = sta_y
                    dset = grp.create_dataset('STX,STY,STZ', (3, ), dtype='f8')
                    dset[:] = xyz

                if not 'STLA,STLO,STDP' in grp.keys():
                    x_map = sta_x * math.sin(az) + sta_y * math.cos(az) + origin_off_xy[0]
                    y_map = sta_x * math.cos(az) - sta_y * math.sin(az) + origin_off_xy[1]
                    sta_lonlat = transformer2lonlat.transform(x_map, y_map)
                    stlalodp[0] = sta_lonlat[1]
                    stlalodp[1] = sta_lonlat[0]
                    dset = grp.create_dataset('STLA,STLO,STDP', (3, ), dtype='f8')
                    dset[:] = stlalodp

                if not 'USEZVALUE' in grp.keys():
                    dset = grp.create_dataset('USEZVALUE', (1, ), dtype='i4')
                    dset[0] = 1

                if not 'X' in grp.keys():
                    grp.create_dataset("X", (npts, ), dtype='f4')

                if not 'Y' in grp.keys():
                    grp.create_dataset("Y", (npts, ), dtype='f4')

                if not 'Z' in grp.keys():
                    grp.create_dataset("Z", (npts, ), dtype='f4')

                if not 'NPTS' in grp.keys():
                    dset = grp.create_dataset('NPTS', (1, ), dtype='i4')
                    dset[0] = 0

                # Check for last sta_y in sta_x, if it has data, can skip everything before
                if grp['NPTS'][0] == npts:
                    last_sta_y = sta_y
            #End for x
            # print(sta_name, grp['NPTS'][0], npts, last_sta_y)
        # End for y
        # End need create
        # outfile.flush()
        outfile.close()
        print('Rank 0: finished file and dset creation, last processed y', last_sta_y, flush=True)

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
    #End mpi rank 0, create template file with all stations and metadata

    last_sta_y = comm.bcast(last_sta_y, root=0)
    # comm.Barrier();

    #Start extracting and writing data
    ssifile = h5py.File(ssi_fname, 'r')

    my_xmin = int(xmin + spacing * mpi_rank)
    my_xmax = int(xmin + spacing * (mpi_rank+1))

    print('Rank', mpi_rank, 'xmin xmax', my_xmin, my_xmax, flush=True)

    df = pd.DataFrame()
    ts = np.linspace(0, delta *(npts - 1), npts)
    df['Time'] = ts.tolist()

    outfile = h5py.File(out_fname, 'r+')

    sta_count = 1
    tic = time.perf_counter()
    if my_xmin <= xmax:
        for sta_x in range(my_xmin, my_xmax, spacing):
            for sta_y in range(last_sta_y, ymax, spacing):

                sta_name = 'S_' + str(sta_x) + '_' + str(sta_y)
                # if sta_name in outfile.keys() and regen_image == False:
                #     if 'NPTS' in outfile[sta_name].keys():
                #         if outfile[sta_name]['NPTS'][0] == npts:
                #             print('skipped', sta_name, flush = True)
                #             continue

                # print('Rank', mpi_rank, sta_name, flush=True)

                for cmpid in range(0,3):

                    vel_name = 'vel_' + str(cmpid) + ' ijk layout'
                    xstart =  int(sta_x / h)
                    ystart =  int(sta_y / h)
                    xend   =  int(sta_x / h + 1)
                    yend   =  int(sta_y / h + 1)

                    data_vel = ssifile[vel_name][ :, xstart:xend, ystart:yend, 0:1 ].flatten()

                    if velocity_sign[cmpid] == -1:
                        data_vel = -data_vel

                    if sta_count % 10 == 0:
                        toc = time.perf_counter()
                        now = datetime.now() # current date and time
                        print('Rank', mpi_rank, ': processed', sta_count, 'stations, until', sta_name,', took', toc - tic, now.strftime("%m/%d/%Y, %H:%M:%S"), flush = True)

                    # debug
                    # print('Rank', mpi_rank, sta_name, vel_name, cmpid, dset_names[cmpid], xstart, xend, ystart, yend, np.min(data_vel), np.max(data_vel), flush=True) 

                    # print('Rank:', mpi_rank, sta_name, dset_names[cmpid], data_vel[int(npts/3)], data_vel[int(npts/2)], data_vel[int(npts-1)])
                    dset = outfile[sta_name][dset_names[cmpid]]
                    dset[:] = data_vel[:]
                    if cmpid == 2:
                        dset = outfile[sta_name]['NPTS']
                        dset[0] = npts
                        outfile.flush()

                    # Write velocity, one writer at a time
                    # if mpi_rank == 0:
                    #     # print('Rank:', mpi_rank, 'open file', flush=True)
                    #     # outfile = h5py.File(out_fname, 'r+', libver='latest', swmr=False)
                    #     outfile = h5py.File(out_fname, 'r+', driver='stdio')
                    #     # print('Rank:', mpi_rank, sta_name, dset_names[cmpid], data_vel[int(npts/3)], data_vel[int(npts/2)], data_vel[int(npts-1)])
                    #     dset = outfile[sta_name][dset_names[cmpid]]
                    #     dset[:] = data_vel[:]
                    #     if cmpid == 2:
                    #         outfile[sta_name]['NPTS'][0] = npts
                    #     outfile.flush()
                    #     outfile.close()
                    #     if mpi_size > 1:
                    #         # print('Rank:', mpi_rank, 'send to 1', flush=True)
                    #         comm.send(mpi_rank, dest=1, tag=11)
                    # else:
                    #     tmp = comm.recv(source=mpi_rank-1, tag=11)
                    #     # print('Rank:', mpi_rank, 'recv from', tmp, ', open file', flush=True)
                    #     # outfile = h5py.File(out_fname, 'r+', libver='latest', swmr=False)
                    #     outfile = h5py.File(out_fname, 'r+', driver='stdio')
                    #     # print('Rank:', mpi_rank, sta_name, dset_names[cmpid], data_vel[int(npts/3)], data_vel[int(npts/2)], data_vel[int(npts-1)])
                    #     dset = outfile[sta_name][dset_names[cmpid]]
                    #     dset[:] = data_vel[:]
                    #     if cmpid == 2:
                    #         outfile[sta_name]['NPTS'][0] = npts
                    #     outfile.flush()
                    #     outfile.close()
                    #     if mpi_rank != mpi_size-1:
                    #         # print('Rank:', mpi_rank, 'send to', mpi_rank+1, flush=True)
                    #         comm.send(mpi_rank, dest=mpi_rank+1, tag=11)

                    # Need to flip the sign for all components
                    data_acc = np.gradient(data_vel, delta, axis=0)
                    data_dis = scipy.integrate.cumtrapz(y=data_vel, dx=delta, initial=0, axis=0)                    
                   
                    vel_image_name = image_dir + '/' + sta_name + '_vel_' + dset_names[cmpid] + '.png'
                    dis_image_name = image_dir + '/' + sta_name + '_dis_' + dset_names[cmpid] + '.png'
                    acc_image_name = image_dir + '/' + sta_name + '_acc_' + dset_names[cmpid] + '.png'
                    csv_name = csv_dir + '/' + sta_name + '.csv'
                    
                    if regen_image or not os.path.exists(csv_name):
                        df['Vel ' + dset_names[cmpid]] = data_vel.tolist()
                        df['Acc ' + dset_names[cmpid]] = data_acc.tolist()
                        df['Dis ' + dset_names[cmpid]] = data_dis.tolist()
                        df.to_csv(csv_name, index=False, float_format='%.6f')
                        
                    if regen_image or not os.path.exists(acc_image_name):
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        plt.subplots_adjust(bottom=0.15)
                        plt.ylim([min(vel_min[cmpid], np.min(data_vel)), max(vel_max[cmpid], np.max(data_vel))])
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=14)
                        line, = ax.plot(ts, data_vel, color='red') 
                        legend = 'min=' + f"{np.min(data_vel):.2f}" + ', max=' + f"{np.max(data_vel):.2f}" 
                        #ax.legend([line],[legend], loc='upper right', fontsize=16)
                        ax.legend([line],[legend], fontsize=16)
                        ax.set_ylabel('Velocity ' + dset_names[cmpid] + ' (m/s)', fontsize=16)
                        ax.set_xlabel('Time (s)', fontsize=16)
                        plt.tight_layout()
                        if save_image:
                            fig.savefig(vel_image_name, dpi=fig_dpi)

                        fig.clf()
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        plt.subplots_adjust(bottom=0.15)
                        plt.ylim([min(dis_min[cmpid], np.min(data_dis)), max(dis_max[cmpid], np.max(data_dis))])
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=14)
                        ax.plot(ts, data_dis, color='red')                
                        legend = 'min=' + f"{np.min(data_dis):.2f}" + ', max=' + f"{np.max(data_dis):.2f}" 
                        #ax.legend([legend], loc='upper right', fontsize=16)
                        ax.legend([legend], fontsize=16)
                        ax.set_ylabel('Displacement ' + dset_names[cmpid] + ' (m)', fontsize=16)
                        ax.set_xlabel('Time (s)', fontsize=16)
                        plt.tight_layout()
                        if save_image:
                            fig.savefig(dis_image_name, dpi=fig_dpi)

                        fig.clf()
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        plt.subplots_adjust(bottom=0.15)
                        plt.ylim([min(acc_min[cmpid], np.min(data_acc)), max(acc_max[cmpid], np.max(data_acc))])
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=14)
                        ax.plot(ts, data_acc, color='red')                
                        legend = 'min=' + f"{np.min(data_acc):.2f}" + ', max=' + f"{np.max(data_acc):.2f}" 
                        #ax.legend([legend], loc='upper right', fontsize=16)
                        ax.legend([legend], fontsize=16)
                        ax.set_ylabel('Acceleration ' + dset_names[cmpid] + ' (m/$s^2$)', fontsize=16)
                        ax.set_xlabel('Time (s)', fontsize=16)
                        plt.tight_layout()
                        if save_image:
                            fig.savefig(acc_image_name, dpi=fig_dpi)
                        fig.clf()
                        plt.close('all')   
                        
                    # del data_vel
                    # del data_acc
                    # del data_dis

		# End for compid
                sta_count += 1
                if sta_count % 10 == 0:
                    tic = time.perf_counter()
	    #End for y
            comm.Barrier()
        #End for x
    # End if my_xmin <= xmax:

    outfile.close()
    ssifile.close()

    comm.Barrier()

    if mpi_rank == 0:
        print('Done!')
