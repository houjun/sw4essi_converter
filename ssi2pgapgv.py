#!/usr/bin/env python
import numpy as np
import h5py
import hdf5plugin
import math
import time
import os
import argparse
import scipy
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="full path to the ssi file",
                        default="")
    parser.add_argument("-o",
                        "--output",
                        help="full path to the output file",
                        default="")
    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")
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
    if args.spacing:
        spacing = args.spacing[0]

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    dset_names = ['X', 'Y', 'Z']

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

    if verbose and mpi_rank == mpi_size - 1:
        print('h', h)
        print('az', az)
        print('origin_lon', origin_lon)
        print('origin_lat', origin_lat)
        print('chk_size', chk_size)
        print('delta, npts, t1', delta, npts, t0 + delta * (npts - 1))
        print('origin', x0, y0, z0)
        print('nx, ny, nz', nx, ny, nz)

    spacing = int(np.ceil(spacing / h)) # 10,000 / 1.75 = 5715
    out_nx = int(np.ceil(nx / spacing)) # 68573 / 5715 = 12
    out_ny = int(np.ceil(ny / spacing)) # 45716 / 5715 = 8

    if mpi_rank == 0:
        print("Start, downsample", spacing, datetime.now())

    # Decompose by x
    my_count = int(np.ceil(out_nx / mpi_size))  # 12 / 16 = 1
    my_start = int(mpi_rank * my_count)
    my_real_count = my_count
    if mpi_size > 1 and mpi_rank == mpi_size - 1:
        my_real_count = out_nx - my_start
    if my_start + my_real_count > out_nx:
        my_real_count = out_nx - my_start
    if my_real_count < 0:
        my_real_count = 0

    my_shape = (my_count, out_ny)               # 1, 8

    # print('Rank', mpi_rank, ', start', my_start, ', count', my_real_count, "outnx, ny", out_nx, out_ny, flush=True)

    my_hpgv = np.zeros(my_shape)
    my_hpga = np.zeros(my_shape)
    my_vpgv = np.zeros(my_shape)
    my_vpga = np.zeros(my_shape)

    if mpi_rank == 0:
        total_shape = (int(my_count * mpi_size), out_ny)
        all_hpgv = np.zeros(total_shape)
        all_hpga = np.zeros(total_shape)
        all_vpgv = np.zeros(total_shape)
        all_vpga = np.zeros(total_shape)
        # print('My shape', my_shape, 'total shape', total_shape)
    else:
        all_hpgv = None
        all_hpga = None
        all_vpgv = None
        all_vpga = None

    #Start extracting and writing data
    i = 0
    ssifile = h5py.File(ssi_fname, 'r')
    for x in range(0, nx, spacing):
        if x >= my_start*spacing and x < (my_start + my_real_count)*spacing:
            j = 0
            for y in range(0, ny, spacing):

                data_vel_0 = ssifile['vel_0 ijk layout'][ :, x, y, : ].flatten()
                data_vel_1 = ssifile['vel_1 ijk layout'][ :, x, y, : ].flatten()
                data_vel_2 = ssifile['vel_2 ijk layout'][ :, x, y, : ].flatten()

                data_acc_0 = np.gradient(data_vel_0, delta, axis=0)
                data_acc_1 = np.gradient(data_vel_1, delta, axis=0)
                data_acc_2 = np.gradient(data_vel_2, delta, axis=0)

                vel_max_0 = np.max(np.absolute(data_vel_0))
                vel_max_1 = np.max(np.absolute(data_vel_1))
                vel_max_2 = np.max(np.absolute(data_vel_2))

                acc_max_0 = np.max(np.absolute(data_acc_0))
                acc_max_1 = np.max(np.absolute(data_acc_1))
                acc_max_2 = np.max(np.absolute(data_acc_2))

                my_hpgv[i,j] = max(vel_max_0, vel_max_1)
                my_vpgv[i,j] = vel_max_2
                my_hpga[i,j] = max(acc_max_0, acc_max_1)
                my_vpga[i,j] = acc_max_2

                # print('Rank', mpi_rank, 'local hpgv, xy =', x, y, 'ij =', i, j, ':', my_hpgv[i,j])

                j+=1
            # End for y
            i+=1
            if i % 5 == 0:
                print('Rank', mpi_rank, 'processed', i, 'x locations', my_real_count, flush=True)
        # End if x in range
    # End for x
    ssifile.close()

    # debug
    # print('Rank', mpi_rank, 'local HPGV 0', my_hpgv[0,:])

    comm.Gather(my_hpgv, all_hpgv, root=0)
    comm.Gather(my_hpga, all_hpga, root=0)
    comm.Gather(my_vpgv, all_vpgv, root=0)
    comm.Gather(my_vpga, all_vpga, root=0)

    if mpi_rank == 0:
        # print('All HPGV 0', all_hpgv[0,:])
        # print('All HPGV 1', all_hpgv[0,:])
        # print('All HPGV 2', all_hpgv[0,:])
        outfile = h5py.File(out_fname, 'w')
        hpgv_dset = outfile.create_dataset('HPGV', data=all_hpgv[0:out_nx, :])
        vpgv_dset = outfile.create_dataset('VPGV', data=all_hpga[0:out_nx, :])
        hpga_dset = outfile.create_dataset('HPGA', data=all_vpgv[0:out_nx, :])
        vpga_dset = outfile.create_dataset('VPGA', data=all_vpga[0:out_nx, :])

        outfile.close()

    comm.Barrier()

    if mpi_rank == 0:
        print("End", datetime.now())
