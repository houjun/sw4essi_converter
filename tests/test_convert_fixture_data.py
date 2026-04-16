import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import h5py
import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "mplconfig-convert-tests"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class _FakeComm:
    def Barrier(self):
        return None

    def Allgather(self, sendbuf, recvbuf):
        recvbuf[0][0] = sendbuf[0][0]

    def send(self, *_args, **_kwargs):
        raise AssertionError("send() should not be called in serial tests")

    def recv(self, *_args, **_kwargs):
        raise AssertionError("recv() should not be called in serial tests")


mpi4py_stub = types.ModuleType("mpi4py")
mpi4py_stub.MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm(), INT=object())
sys.modules.setdefault("mpi4py", mpi4py_stub)
sys.modules.setdefault("hdf5plugin", types.ModuleType("hdf5plugin"))

import convert


class ConvertFixtureDataTests(unittest.TestCase):
    def setUp(self):
        convert.MPI = mpi4py_stub.MPI
        self.sample_ssi = REPO_ROOT / "tests" / "data" / "small.ssi"
        self.sample_h5 = REPO_ROOT / "template" / "h5NodeCrds.h5"
        self.sample_csv = REPO_ROOT / "template" / "motion_setting.csv"
        self.reference_output = REPO_ROOT / "tests" / "data" / "h5NodeMotion.h5"

    def test_convert_h5_matches_checked_in_fixture_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_coord, start_t, end_t, tstep, rotate_angle, zero_motion_dir = convert.get_csv_meta(
                str(self.sample_csv)
            )

            convert.convert_h5(
                str(self.sample_h5),
                str(self.sample_ssi),
                tmpdir,
                ref_coord,
                start_t,
                end_t,
                tstep,
                rotate_angle,
                zero_motion_dir,
                False,
                0,
                1,
                False,
            )

            output_path = Path(tmpdir) / "h5NodeCrds_motion.h5"
            self.assertTrue(output_path.exists(), f"Missing output file: {output_path}")

            with h5py.File(output_path, "r") as output_h5, h5py.File(self.reference_output, "r") as reference_h5:
                self.assertEqual(output_h5["velocity"].shape, (108, 600))
                self.assertEqual(output_h5["displacement"].shape, (108, 600))
                np.testing.assert_allclose(output_h5["acceleration"][:], reference_h5["acceleration"][:])
                np.testing.assert_allclose(output_h5["xyz"][:], reference_h5["xyz"][:])
                np.testing.assert_array_equal(output_h5["nodeTag"][:], reference_h5["nodeTag"][:])
                self.assertEqual(float(output_h5["dt"][()]), float(reference_h5["dt"][()]))
                self.assertEqual(float(output_h5["tstart"][()]), float(reference_h5["tstart"][()]))

                expected_tend = float(output_h5["dt"][()]) * (output_h5["acceleration"].shape[1] - 1)
                self.assertAlmostEqual(float(output_h5["tend"][()]), expected_tend)


if __name__ == "__main__":
    unittest.main()
