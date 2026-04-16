import sys
import tempfile
import types
import unittest
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

mpi4py_stub = types.ModuleType("mpi4py")
mpi4py_stub.MPI = types.SimpleNamespace()
sys.modules.setdefault("mpi4py", mpi4py_stub)
sys.modules.setdefault("hdf5plugin", types.ModuleType("hdf5plugin"))

import convert


class ConvertHelperTests(unittest.TestCase):
    def test_get_flat_coord_range_scales_node_offsets_by_xyz_width(self):
        self.assertEqual(convert.get_flat_coord_range(0, 2), (0, 6))
        self.assertEqual(convert.get_flat_coord_range(2, 3), (6, 15))

    def test_create_hdf5_essi_writes_float_time_axis_matching_motion_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "template.h5"
            with h5py.File(path, "w"):
                pass

            convert.create_hdf5_essi(
                str(path),
                ncoord=2,
                nstep=3,
                dt=0.25,
                gen_vel=True,
                gen_acc=True,
                gen_dis=True,
                extra_dname="unused",
            )

            with h5py.File(path, "r") as h5file:
                self.assertEqual(h5file["Velocity"].shape, (6, 3))
                self.assertEqual(h5file["Accelerations"].shape, (6, 3))
                self.assertEqual(h5file["Displacements"].shape, (6, 3))
                self.assertEqual(h5file["Time"].shape, (3,))
                self.assertEqual(h5file["Time"].dtype, np.dtype("float64"))
                np.testing.assert_allclose(h5file["Time"][:], [0.0, 0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
