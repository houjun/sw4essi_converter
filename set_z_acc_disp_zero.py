import sys
import h5py
import numpy as np

drm_filename = sys.argv[1]
drm_file = h5py.File(drm_filename)
coordinates = drm_file['Coordinates']
n_coord = int(coordinates.shape[0] / 3)

nt = drm_file['Time'].shape[0]

start = 0
for i in range(start, nt):
    print('Iter', i, 'n_coord', n_coord)
    for j in range(0, n_coord):
        drm_file['Accelerations'][j*3+2, i] = 0 
        drm_file['Displacements'][j*3+2, i] = 0


drm_file.close()
