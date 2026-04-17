# GEOtoENG

GEOtoENG converts SW4 SSI output into motion files that can be used by OpenSees and related workflows.

The script reads an SSI HDF5 file, maps user node coordinates into the SW4 grid, optionally rotates or zeroes motion components, and writes truncated motion histories. ZFP-compressed SSI files are supported through `hdf5plugin`.

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
- `--csv` with `--template`: use a CSV mapping file together with an SSI template file

When `--output-format` is omitted, the script keeps the current default for each input mode:
- DRM input defaults to OpenSees DRM output
- HDF5 and CSV inputs default to point-motion HDF5 output
- CSV + template input defaults to ESSI template output

## Main Arguments

```text
-c,  --csv            CSV settings / coordinate file
-d,  --drm            DRM file with node coordinates
-h5, --hdf5           HDF5 file with node coordinates
-e,  --ssi            SW4 SSI output file
-t,  --template       SSI template file
-o,  --output-format  point, opensees, or essi
-P,  --savepath       output directory
-r,  --reference      reference coordinate offset
-R,  --rotateanlge    rotation angle in degrees
-s,  --timerange      start end step
-z,  --zeroMotionDir  None, x, y, or z
-p,  --plotonly       only plot input nodes
-v,  --verbose        verbose logging
```

Notes:
- `--timerange start end step` is in SSI time units.
- If `start == end`, the script keeps all steps from `start` to the end of the SSI record.
- `--output-format point` is supported for DRM, HDF5, CSV, and template-driven inputs.
- `--output-format opensees` and `--output-format essi` require input that carries boundary-node metadata, so they are only supported for DRM or template-driven inputs.

## Examples

Generate OpenSees DRM input from the checked-in sample files:

```bash
mpirun -np 3 python convert.py \
  -d template/DRMTemplate.h5drm \
  --ssi tests/data/small.ssi \
  -c template/motion_setting.csv \
  -P tests/
```

Generate motion histories from an HDF5 node file:

```bash
mpirun -np 3 python convert.py \
  -h5 template/h5NodeCrds.h5 \
  --ssi tests/data/small.ssi \
  -c template/motion_setting.csv \
  -P tests/
```

Force point-motion output explicitly from an HDF5 node file:

```bash
mpirun -np 3 python convert.py \
  -h5 template/h5NodeCrds.h5 \
  --ssi tests/data/small.ssi \
  -c template/motion_setting.csv \
  --output-format point \
  -P tests/
```

Generate OpenSees DRM output from a template-driven run:

```bash
mpirun -np 3 python convert.py \
  -c template/motion_setting.csv \
  -t template/DRMTemplate.h5drm \
  --ssi tests/data/small.ssi \
  --output-format opensees \
  -P tests/
```

Generate plots only:

```bash
mpirun -np 1 python convert.py \
  -d template/DRMTemplate.h5drm \
  --ssi tests/data/small.ssi \
  -c template/motion_setting.csv \
  -p
```

## Checked-in Sample Files

- `template/DRMTemplate.h5drm`
- `template/h5NodeCrds.h5`
- `template/motion_setting.csv`
- `tests/data/small.ssi`
