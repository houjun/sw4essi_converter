# SW4 ESSI Converter

Convert SW4 ESSI output into motion files that can be used by OpenSees and related workflows.

The script reads an ESSI HDF5 file, maps user node coordinates into the SW4 grid, optionally rotates or zeroes motion components, and writes truncated motion histories. ZFP-compressed ESSI files are supported through `hdf5plugin`.

## Requirements

Python packages:
- `numpy`
- `scipy`
- `h5py`
- `mpi4py`
- `matplotlib`
- `hdf5plugin`

Typical run pattern:

```bash
mpirun -np 4 python convert.py ...
```

## Input Modes

- `--drm`: read node coordinates from an OpenSees DRM template and write `DRMinput.h5drm`
- `--hdf5`: read node coordinates from an HDF5 file and write `h5NodeMotion.h5`
- `--csv`: read node coordinates from a CSV file and write `csvNodeMotion.h5`
- `--csv` with `--template`: use a CSV mapping file together with an ESSI template file

## Main Arguments

```text
-c,  --csv            CSV settings / coordinate file
-d,  --drm            DRM file with node coordinates
-h5, --hdf5           HDF5 file with node coordinates
-e,  --essi           SW4 ESSI output file
-t,  --template       ESSI template file
-P,  --savepath       output directory
-r,  --reference      reference coordinate offset
-R,  --rotateanlge    rotation angle in degrees
-s,  --timerange      start end step
-z,  --zeroMotionDir  None, x, y, or z
-p,  --plotonly       only plot input nodes
-v,  --verbose        verbose logging
```

Notes:
- `--timerange start end step` is in ESSI time units.
- If `start == end`, the script keeps all steps from `start` to the end of the ESSI record.

## Examples

Generate OpenSees DRM input from the checked-in sample files:

```bash
mpirun -np 3 python convert.py \
  -d template/DRMTemplate.h5drm \
  -e test/small.essi \
  -c template/motion_setting.csv \
  -P test/
```

Generate motion histories from an HDF5 node file:

```bash
mpirun -np 3 python convert.py \
  -h5 template/h5NodeCrds.h5 \
  -e test/small.essi \
  -c template/motion_setting.csv \
  -P test/
```

Generate plots only:

```bash
mpirun -np 1 python convert.py \
  -d template/DRMTemplate.h5drm \
  -e test/small.essi \
  -c template/motion_setting.csv \
  -p
```

## Checked-in Sample Files

- `template/DRMTemplate.h5drm`
- `template/h5NodeCrds.h5`
- `template/motion_setting.csv`
- `test/small.essi`
