import sys
import h5py
import numpy as np
import math


drm_filename = sys.argv[1]
drm_file = h5py.File(drm_filename, 'r')
coordinates = drm_file['Coordinates']
n_coord = int(coordinates.shape[0] / 3)

Time_dset          = drm_file['Time'][()]
Accelerations_dset = drm_file['Accelerations'][()]
Displacements_dset = drm_file['Displacements'][()]

y_bot = 0
y_top = 120
x_loc = 10

iter = 1

is_all_same = True

# Store the coordiates in individual x, y, z arrays
for i in range (0, n_coord):
    
    drm_x = coordinates[i*3]
    drm_y = coordinates[i*3+1]
    drm_z = coordinates[i*3+2]

    loc = 3 * i
    x_acc  = Accelerations_dset[loc,:]
    y_acc  = Accelerations_dset[loc+1,:]
    z_acc  = Accelerations_dset[loc+2,:]
    x_disp = Displacements_dset[loc,:]
    y_disp = Displacements_dset[loc+1,:]
    z_disp = Displacements_dset[loc+2,:]

    for j in range (i+1, n_coord):
        drm_x2 = coordinates[j*3]
        drm_y2 = coordinates[j*3+1]
        drm_z2 = coordinates[j*3+2]

        if drm_x == drm_x2 and drm_y == drm_y2:
            print('checking', i, '(', drm_x, drm_y, drm_z, ')', j , '(', drm_x2, drm_y2, drm_z2, ')')

            loc = 3 * j
            x_acc2  = Accelerations_dset[loc,:]
            y_acc2  = Accelerations_dset[loc+1,:]
            z_acc2  = Accelerations_dset[loc+2,:]
            x_disp2 = Displacements_dset[loc,:]
            y_disp2 = Displacements_dset[loc+1,:]
            z_disp2 = Displacements_dset[loc+2,:]
            
            if not np.array_equal(x_acc, x_acc2):
                print('Have different x acc values!')
                is_all_same = False
            if not np.array_equal(y_acc, y_acc2):
                print('Have different y acc values!')
                is_all_same = False

            if not np.array_equal(z_acc, z_acc2):
                print('Have different z acc values!')
                is_all_same = False
                
            if not np.array_equal(x_disp, x_disp2):
                print('Have different x disp values!')
                is_all_same = False

            if not np.array_equal(y_disp, y_disp2):
                print('Have different y disp values!')
                is_all_same = False

            if not np.array_equal(z_disp, z_disp2):
                print('Have different z disp values!')
                is_all_same = False
            
            if is_all_same:
                print('all values match')
            else:
                sys.exit()
