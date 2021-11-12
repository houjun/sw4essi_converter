# SW4 ESSI converter
This code converts the SW4 SSIoutput (w/ ZFP compression) to be used by analysis codes such as OpenSees.
It requires scipy, numpy, h5py, mpi4py, hdf5plugin (for ZFP), matplotlib python packages.

Supported command line arguments:
```
-c or --csv: full path to the CSV setting file, default=""
-d or --drm: full path to the OpenSees DRM template file with node coordinates, default=""
-e or --essi: full path to the SW4 ESSI output file, default=""
-t or --template: full path to the ESSI template file with node coordinates, default=""
-p or --plotonly: only generate plots of the input nodes
-P or --savepath: absolute or relative path for saving the result files
-r or --reference: reference node coordinate offset, default 0 0 0
-R or --rotateanlge: rotate angle for node coordinate and motion: [0, 360)
-s or --timerange: time range, will return all steps after the lower limit for equal upper and lower limit"
-v or --verbose: increase output verbosity
-z or --zeroMotionDir: direction for zeroing out motion and enforce same motion across nodes in that direction: None(default), x, y, z
```

Example usage:
```
OpenSees output:
> mpirun -np 3 python convert.py -d template/DRMTemplate.h5drm -e test/small.essi -c template/motion_setting.csv -P test/
ESSI output:
> mpirun -np 16 python convert.py -c Parameters_for_motion_conversion.csv -e Location53.essi -t Cubic_200_template.hdf5 
```
