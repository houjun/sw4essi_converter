# SW4 ESSI converter
This code converts the SW4 SSIoutput (w/ ZFP compression) to be used by analysis codes such as OpenSees.
It requires scipy, numpy, h5py, mpi4py, hdf5plugin (for ZFP), matplotlib python packages.

Supported command line arguments:
```
-c or --csv: full path to the CSV setting file, default=""
-d or --drm: full path to the DRM file with node coordinates, default=""
-e or --essi: full path to the SW4 ESSI output file, default=""
-p or --plotonly: only generate plots of the input nodes
-r or --reference: reference node coordinate offset, default 0 0 0
-t or --trange: timestep range, default 0 total_steps
-v or --verbose: increase output verbosity
```

Example usage:
```
> mpirun -np 16 python converter.py -d OpenSeesDRMTemplate.h5drm -e Location1.essi
```
