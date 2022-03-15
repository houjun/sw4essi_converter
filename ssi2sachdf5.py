#!/usr/bin/env python

import numpy as np
import h5py
import hdf5plugin
from pyproj import CRS
from pyproj import Transformer
import math
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import argparse
matplotlib.rcParams['figure.dpi'] = 150


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
        if not sta_name in outfile.keys():
            for sta_x in range(xmin, xmax, spacing):
                for sta_y in range(ymin, ymax, spacing):
                    sta_name = 'S_' + str(sta_x) + '_' + str(sta_y)
                    x_map = sta_x * math.sin(az) + sta_y * math.cos(az) + origin_off_xy[0]
                    y_map = sta_x * math.cos(az) - sta_y * math.sin(az) + origin_off_xy[1]
                    sta_lonlat = transformer2lonlat.transform(x_map, y_map)

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
                        stlalodp[0] = sta_lonlat[1]
                        stlalodp[1] = sta_lonlat[0]
                        dset = grp.create_dataset('STLA,STLO,STDP', (3, ),
                                                  dtype='f8')
                        dset[:] = stlalodp

                    if not 'USEZVALUE' in grp.keys():
                        dset = grp.create_dataset('USEZVALUE', (1, ), dtype='i4')
                        dset[0] = 1

                    if not 'X' in grp.keys():
                        dset = grp.create_dataset("X", (npts, ), dtype='f4')

                    if not 'Y' in grp.keys():
                        dset = grp.create_dataset("Y", (npts, ), dtype='f4')

                    if not 'Z' in grp.keys():
                        dset = grp.create_dataset("Z", (npts, ), dtype='f4')

                    if not 'NPTS' in grp.keys():
                        dset = grp.create_dataset('NPTS', (1, ), dtype='i4')
                        dset[0] = 0

            #End for station
        # End need create
        outfile.flush()
        outfile.close()

    #End mpi rank 0, create template file with all stations and metadata
    comm.Barrier();

    #Start extracting and writing data
    ssifile = h5py.File(ssi_fname, 'r')
    outfile = h5py.File(out_fname, 'a')

    my_xmin = int(xmin + spacing * mpi_rank)
    my_xmax = int(xmin + spacing * (mpi_rank+1))

    print('Rank', mpi_rank, 'xmin xmax', my_xmin, my_xmax, flush=True)

    sta_count = 1

    tic = time.perf_counter()
    if my_xmin <= xmax:
        for sta_x in range(my_xmin, my_xmax, spacing):
            for sta_y in range(ymin, ymax, spacing):

                sta_name = 'S_' + str(sta_x) + '_' + str(sta_y)
                x_map = sta_x * math.sin(az) + sta_y * math.cos(az) + origin_off_xy[0]
                y_map = sta_x * math.cos(az) - sta_y * math.sin(az) + origin_off_xy[1]
                sta_lonlat = transformer2lonlat.transform(x_map, y_map)

                # print(sta_name, sta_lonlat, flush = True)

                if sta_name in outfile.keys() :
                    if 'NPTS' in outfile[sta_name].keys() :
                        if outfile[sta_name]['NPTS'][0] == npts:
                            print('skipped', sta_name, flush = True)
                            continue

                data_x = ssifile['vel_0 ijk layout'][ :, int(sta_x / h) : int(sta_x / h + 1), int(sta_y / h) : int(sta_y / h + 1), 0 : 1].flatten()
                data_y = ssifile['vel_1 ijk layout'][ :, int(sta_x / h) : int(sta_x / h + 1), int(sta_y / h) : int(sta_y / h + 1), 0 : 1].flatten()
                data_z = ssifile['vel_2 ijk layout'][ :, int(sta_x / h) : int(sta_x / h + 1), int(sta_y / h) : int(sta_y / h + 1), 0 : 1].flatten()
                #print(len(data_x), data_x)
                if sta_count % 10 == 0:
                    toc = time.perf_counter()
                    now = datetime.now() # current date and time
                    print('Rank', mpi_rank, ': processed', sta_count, 'stations, until', sta_name,', took', toc - tic, now.strftime("%m/%d/%Y, %H:%M:%S"), flush = True)

                # #ts = np.linspace(0, delta *(npts - 1), npts)
                # #plt.plot(ts, data_x.flatten(), '-', color = 'red', label = 'Vsmin=140')

                grp = outfile[sta_name]

                dset = grp['X']
                dset[ : ] = data_x
                dset = grp['Y']
                dset[ : ] = data_y
                dset = grp['Z']
                dset[ : ] = data_z

                dset = grp['NPTS']
                dset[0] = npts

                outfile.flush()

                del data_x
                del data_y
                del data_z
                sta_count += 1

                if sta_count % 10 == 0:
                    tic = time.perf_counter()
        #End for station

    ssifile.close()
    outfile.close()

    comm.Barrier()

    if mpi_rank == 0:
        print('Done!')
