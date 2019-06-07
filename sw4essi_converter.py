# import os
import sys
import h5py
import numpy as np
import datetime
# import matplotlib.pyplot as plt
import math

def interpolate(val0, val1, fraction):
    return val0 + (val1 - val0) * fraction

def main():
    print('Start time:', datetime.datetime.now().time())


    # drm_filename = 'DRM_layer_geometric_info.hdf5'
    # start_x = -88.392
    # start_y = 12.802
    # start_y = 12.8016

    # drm_filename = 'DRM_layer_geometric_info_Test.hdf5'
    drm_filename = sys.argv[1]
    print('Input file:', drm_filename)
    start_x = 0
    start_y = 120

    sw4essiout_filename = 'M6.0_ESSI_SRF_X.cycle=00000.essi'

    # Get the coordinates from DRM file
    drm_file = h5py.File(drm_filename)
    coordinates = drm_file['Coordinates']
    n_coord = int(coordinates.shape[0] / 3)
    print('# of coordinates: ', n_coord)
    drm_x = np.zeros(n_coord)
    drm_y = np.zeros(n_coord)
    drm_z = np.zeros(n_coord)

    # Store the coordiates in individual x, y, z arrays
    for i in range(0, n_coord):
        drm_x[i] = coordinates[i*3]
        drm_y[i] = coordinates[i*3+1]
        drm_z[i] = coordinates[i*3+2]

    print('First xyz: ', drm_x[0], drm_y[0], drm_z[0])
    print('Last xyz: ', drm_x[n_coord-1], drm_y[n_coord-1], drm_z[n_coord-1])

    # Get parameter values from HDF5 data
    sw4essiout = h5py.File(sw4essiout_filename, 'r')
    h  = sw4essiout['ESSI xyz grid spacing'][0]
    x0 = sw4essiout['ESSI xyz origin'][0]
    y0 = sw4essiout['ESSI xyz origin'][1]
    z0 = sw4essiout['ESSI xyz origin'][2]
    t0 = sw4essiout['time start'][0]
    npts = sw4essiout['cycle start, end'][1]
    dt = sw4essiout['timestep'][0]
    t1 = dt*(npts-1)
    print ('grid spacing, h: ', h)
    print ('ESSI origin x0, y0, z0: ', x0, y0, z0)
    print('timing, t0, dt, npts, t1: ', t0, round(dt,6), npts, round(t1,6) )
    print('Shape of HDF5 data: ', sw4essiout['vel_0 ijk layout'].shape)

    nx = sw4essiout['vel_0 ijk layout'].shape[0]
    # ny = sw4essiout['vel_0 ijk layout'].shape[1]
    nz = sw4essiout['vel_0 ijk layout'].shape[2]
    nt = sw4essiout['vel_0 ijk layout'].shape[3]


    time = np.linspace(t0, t1, npts+1)

    if not "Time" in drm_file.keys():
        Time_dset          = drm_file.create_dataset("Time", data=time)
        Accelerations_dset = drm_file.create_dataset("Accelerations", (n_coord*3, nt))
        Displacements_dset = drm_file.create_dataset("Displacements", (n_coord*3, nt))
    else:
        Time_dset          = drm_file['Time']
        Accelerations_dset = drm_file['Accelerations']
        Displacements_dset = drm_file['Displacements']


    yplane_x_vel_all = sw4essiout['vel_0 ijk layout']
    yplane_y_vel_all = sw4essiout['vel_2 ijk layout']
    yplane_z_vel_all = sw4essiout['vel_1 ijk layout']

    x_disp      = np.zeros(shape=(nx,nz))
    y_disp      = np.zeros(shape=(nx,nz))
    z_disp      = np.zeros(shape=(nx,nz))
    prev_x_disp = np.zeros(shape=(nx,nz))
    prev_y_disp = np.zeros(shape=(nx,nz))
    prev_z_disp = np.zeros(shape=(nx,nz))

    yplane = 0
    # nt = 10001
    # for i in range(10000, nt):
    start = 1
    # start = 1000
    for i in range(start, nt-1):

        print('Iter ', i)

        # Get the yplane for each time step
        # yplane_x_vel = sw4essiout['vel_0 ijk layout'][:,yplane,:,i]
        # yplane_y_vel = sw4essiout['vel_1 ijk layout'][:,yplane,:,i]
        # yplane_z_vel = sw4essiout['vel_2 ijk layout'][:,yplane,:,i]

        yplane_x_vel = yplane_x_vel_all[:,yplane,:,i]
        yplane_y_vel = -yplane_y_vel_all[:,yplane,:,i]
        yplane_z_vel = yplane_z_vel_all[:,yplane,:,i]

        prev_yplane_x_vel = yplane_x_vel_all[:,yplane,:,i-1]
        prev_yplane_y_vel = -yplane_y_vel_all[:,yplane,:,i-1]
        prev_yplane_z_vel = yplane_z_vel_all[:,yplane,:,i-1]

        next_yplane_x_vel = yplane_x_vel_all[:,yplane,:,i+1]
        next_yplane_y_vel = -yplane_y_vel_all[:,yplane,:,i+1]
        next_yplane_z_vel = yplane_z_vel_all[:,yplane,:,i+1]

        count = np.count_nonzero(yplane_x_vel)
        if count == 0:
            print('x vel all zeros!')
        else:
            print('x vel non zero count ', count)

        count = np.count_nonzero(yplane_y_vel)
        if count == 0:
            print('y vel all zeros!')
        else:
            print('x vel non zero count ', count)

        count = np.count_nonzero(yplane_z_vel)
        if count == 0:
            print('z vel all zeros!')
        else:
            print('z vel non zero count ', count)

        if i > 1:
            prev_x_disp = x_disp
            prev_y_disp = y_disp
            prev_z_disp = z_disp

        # Acc(i)=(Vel(i+1)-Vel(i-1))/(2*dt);
        x_acc  = (next_yplane_x_vel - prev_yplane_x_vel) / (2.0 * dt)
        # Disp(i)=Disp(i-1)+0.5*(Vel(i-1)+Vel(i))*dt;
        x_disp = prev_x_disp + 0.5 * (prev_yplane_x_vel + yplane_x_vel) * dt

        y_acc  = (next_yplane_y_vel - prev_yplane_y_vel) / (2.0 * dt)
        y_disp = prev_y_disp + 0.5 * (prev_yplane_y_vel + yplane_y_vel) * dt

        z_acc  = (next_yplane_z_vel - prev_yplane_z_vel) / (2.0 * dt)
        z_disp = prev_z_disp + 0.5 * (prev_yplane_z_vel + yplane_z_vel) * dt

        # Calculate the acceleration and displacement for each DRM coordinate
        for j in range(0, n_coord):
            # Find the four surrounding points' index in sw4essiouput data
            x_diff = drm_x[j] - start_x
            y_diff = start_y - drm_y[j]
            if x_diff < 0:
                print('x_diff = ', x_diff)
                x_diff = -x_diff
            if y_diff < 0:
                print('y_diff = ', y_diff)
                y_diff = -y_diff
            
            # z_diff = drm_z[j] - start_z
            x_idx_lo = int(math.floor(x_diff / h))
            x_idx_hi = int(math.ceil(x_diff / h))
            y_idx_lo = int(math.floor(y_diff / h))
            y_idx_hi = int(math.ceil(y_diff / h))

            # Get the left and top coordinate in DRM model
            left_x  = x_idx_lo * h + start_x
            top_y   = y_idx_lo * h 

            x_interp_frac = (drm_x[j] - left_x) / h
            y_interp_frac = (start_y - drm_y[j] - top_y)  / h

            # Top interpolation
            if x_idx_lo >= nx:
                print('x_idx_lo overflow!', x_idx_lo, '/', nx, 'x_diff=', x_diff)
                x_idx_lo = nx - 1
            if y_idx_lo >= nz:
                print('y_idx_lo overflow!', y_idx_lo, '/', nz, 'y_diff=', y_diff)
                y_idx_lo = nz - 1
            if x_idx_hi >= nx:
                print('x_idx_hi overflow!', x_idx_hi, '/', nx, 'x_diff=', x_diff)
                x_idx_hi = nx - 1
            if y_idx_hi >= nz:
                print('y_idx_hi overflow!', y_idx_hi, '/', nz, 'y_diff=', y_diff, 'drm_y=', drm_y[j])
                y_idx_hi = nz - 1

            if x_idx_lo < 0:
                print('x_idx_lo < 0', 'drm_x=', drm_x[j], 'start_x=', start_x, x_diff, x_idx_lo)
            if y_idx_lo < 0:
                print('y_idx_lo < 0', 'drm_y=', drm_y[j], 'start_y=', start_y, y_diff, y_idx_lo)
            if x_idx_hi < 0:
                print('x_idx_hi < 0', 'drm_x=', drm_x[j], 'start_x=', start_x, x_diff, x_idx_hi)
            if y_idx_hi < 0:
                print('y_idx_hi < 0', 'drm_y=', drm_y[j], 'start_y=', start_y, y_diff, y_idx_hi)


            x_top_acc_interp  = interpolate(x_acc[x_idx_lo, y_idx_lo],   x_acc[x_idx_hi, y_idx_lo], x_interp_frac)
            x_top_disp_interp = interpolate(x_disp[x_idx_lo, y_idx_lo], x_disp[x_idx_hi, y_idx_lo], x_interp_frac)

            y_top_acc_interp  = interpolate(y_acc[x_idx_lo, y_idx_lo],   y_acc[x_idx_hi, y_idx_lo], x_interp_frac)
            y_top_disp_interp = interpolate(y_disp[x_idx_lo, y_idx_lo], y_disp[x_idx_hi, y_idx_lo], x_interp_frac)

            z_top_acc_interp  = interpolate(z_acc[x_idx_lo, y_idx_lo],   z_acc[x_idx_hi, y_idx_lo], x_interp_frac)
            z_top_disp_interp = interpolate(z_disp[x_idx_lo, y_idx_lo], z_disp[x_idx_hi, y_idx_lo], x_interp_frac)

            # Bottom interpolation
            x_bot_acc_interp  = interpolate(x_acc[x_idx_hi, y_idx_lo],  x_acc[x_idx_hi, y_idx_hi],  x_interp_frac)
            x_bot_disp_interp = interpolate(x_disp[x_idx_hi, y_idx_lo], x_disp[x_idx_hi, y_idx_hi], x_interp_frac)

            y_bot_acc_interp  = interpolate(y_acc[x_idx_hi, y_idx_lo],  y_acc[x_idx_hi, y_idx_hi],  x_interp_frac)
            y_bot_disp_interp = interpolate(y_disp[x_idx_hi, y_idx_lo], y_disp[x_idx_hi, y_idx_hi], x_interp_frac)

            z_bot_acc_interp  = interpolate(z_acc[x_idx_hi, y_idx_lo],  z_acc[x_idx_hi, y_idx_hi],  x_interp_frac)
            z_bot_disp_interp = interpolate(z_disp[x_idx_hi, y_idx_lo], z_disp[x_idx_hi, y_idx_hi], x_interp_frac)

            # Left and right interpolation
            x_acc_interp  = interpolate(x_top_acc_interp,  x_bot_acc_interp,  y_interp_frac)
            x_disp_interp = interpolate(x_top_disp_interp, x_bot_disp_interp, y_interp_frac)

            y_acc_interp  = interpolate(y_top_acc_interp,  y_bot_acc_interp,  y_interp_frac)
            y_disp_interp = interpolate(y_top_disp_interp, y_bot_disp_interp, y_interp_frac)

            z_acc_interp  = interpolate(z_top_acc_interp,  z_bot_acc_interp,  y_interp_frac)
            z_disp_interp = interpolate(z_top_disp_interp, z_bot_disp_interp, y_interp_frac)

            # Populate the result array for accelerations and displacements
            Accelerations_dset[j*3, i]   = x_acc_interp
            Accelerations_dset[j*3+1, i] = y_acc_interp
            Accelerations_dset[j*3+2, i] = 0 

            Displacements_dset[j*3, i]   = x_disp_interp
            Displacements_dset[j*3+1, i] = y_disp_interp
            Displacements_dset[j*3+2, i] = 0
        #End for each DRM coordinate

        # tmp_acc = Accelerations_dset[:,i]
        # tmp_disp = Displacements_dset[:,i]
        print('Min     acceleration: ', np.min(Accelerations_dset[:,i]))
        print('Average acceleration: ', np.average(Accelerations_dset[:,i]))
        print('Max     acceleration: ', np.max(Accelerations_dset[:,i]))

        print('Min     displacement: ', np.min(Displacements_dset[:,i]))
        print('Average displacement: ', np.average(Displacements_dset[:,i]))
        print('Max     displacement: ', np.max(Displacements_dset[:,i]))
        # print('Shape of plane: ', yplane_x_vel.shape)
        # print(i, ': x values', yplane_x_vel[1,1], yplane_x_vel[5,5])
        sys.stdout.flush()
    # End for each timestep

    sw4essiout.close()
    drm_file.close()
    print('End time:', datetime.datetime.now().time())

main()
