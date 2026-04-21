import csv
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

    def _write_h5_based_csv_fixture(self, output_path):
        ref_coord, start_t, end_t, tstep, rotate_angle, zero_motion_dir = convert.get_csv_meta(
            str(self.sample_csv)
        )

        with h5py.File(self.sample_h5, "r") as input_h5:
            coordinates = input_h5["coordinate"][:]
            node_tags = input_h5["nodeTag"][:]

        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 3)
        elif coordinates.ndim == 2 and coordinates.shape[1] == 1:
            coordinates = coordinates.reshape(-1, 3)

        with output_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "nodeTag",
                    "x",
                    "y",
                    "z",
                    "essiXstart",
                    "essiYstart",
                    "essiZstart",
                    "startTime",
                    "endTime",
                    "tstep",
                    "rotationAngle",
                    "zeroMotionDir",
                ]
            )
            for index, (node_tag, coord) in enumerate(zip(node_tags, coordinates)):
                row = [int(node_tag), float(coord[0]), float(coord[1]), float(coord[2]), "", "", "", "", "", "", "", ""]
                if index == 0:
                    row[4:12] = [ref_coord[0], ref_coord[1], ref_coord[2], start_t, end_t, tstep, rotate_angle, zero_motion_dir]
                writer.writerow(row)


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

    def test_convert_h5_accepts_explicit_point_output_mode(self):
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
                requested_output_mode="point",
            )

            output_path = Path(tmpdir) / "h5NodeCrds_motion.h5"
            self.assertTrue(output_path.exists(), f"Missing output file: {output_path}")

            with h5py.File(output_path, "r") as output_h5, h5py.File(self.reference_output, "r") as reference_h5:
                np.testing.assert_allclose(output_h5["acceleration"][:], reference_h5["acceleration"][:])
                np.testing.assert_allclose(output_h5["xyz"][:], reference_h5["xyz"][:])
                np.testing.assert_array_equal(output_h5["nodeTag"][:], reference_h5["nodeTag"][:])

    def test_convert_h5_accepts_explicit_essi_output_mode(self):
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
                requested_output_mode="essi",
            )

            output_path = Path(tmpdir) / "h5NodeCrds.h5"
            self.assertTrue(output_path.exists(), f"Missing output file: {output_path}")

            with h5py.File(output_path, "r") as output_h5, h5py.File(self.reference_output, "r") as reference_h5:
                self.assertEqual(output_h5["Coordinates"].shape, (108,))
                self.assertEqual(output_h5["Velocity"].shape, (108, 600))
                self.assertEqual(output_h5["Accelerations"].shape, (108, 600))
                self.assertEqual(output_h5["Displacements"].shape, (108, 600))
                self.assertEqual(output_h5["Time"].shape, (600,))
                np.testing.assert_array_equal(output_h5["nodeTag"][:], reference_h5["nodeTag"][:])
                timestep = float(output_h5["Time"][1] - output_h5["Time"][0])
                np.testing.assert_allclose(output_h5["Time"][:], np.arange(600, dtype="f8") * timestep)

    def test_convert_h5_accepts_explicit_opensees_output_mode(self):
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
                requested_output_mode="opensees",
            )

            output_path = Path(tmpdir) / "OpenSeesDRMinput.h5drm"
            self.assertTrue(output_path.exists(), f"Missing output file: {output_path}")

            with h5py.File(output_path, "r") as output_h5, h5py.File(self.reference_output, "r") as reference_h5:
                self.assertEqual(output_h5["DRM_Data"]["acceleration"].shape, (108, 600))
                self.assertEqual(output_h5["DRM_Data"]["displacement"].shape, (108, 600))
                self.assertEqual(output_h5["DRM_Data"]["nodeTag"].shape, (36,))
                self.assertEqual(output_h5["DRM_Data"]["xyz"].shape, (36, 3))
                self.assertEqual(
                    float(output_h5["DRM_Metadata"]["dt"][()]),
                    float(reference_h5["dt"][()]),
                )

    def test_convert_csv_accepts_explicit_essi_output_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "input.csv"
            self._write_h5_based_csv_fixture(csv_path)
            ref_coord, start_t, end_t, tstep, rotate_angle, zero_motion_dir = convert.get_csv_meta(str(csv_path))

            convert.convert_csv(
                str(csv_path),
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
                requested_output_mode="essi",
            )

            output_path = Path(tmpdir) / "input.csv"
            self.assertTrue(output_path.exists(), f"Missing source CSV fixture: {output_path}")
            essi_output = Path(tmpdir) / "input.h5"
            self.assertTrue(essi_output.exists(), f"Missing output file: {essi_output}")

            with h5py.File(essi_output, "r") as output_h5:
                self.assertEqual(output_h5["Coordinates"].shape, (108,))
                self.assertEqual(output_h5["nodeTag"].shape, (36,))
                self.assertEqual(output_h5["Velocity"].shape, (108, 600))
                self.assertEqual(output_h5["Time"].shape, (600,))


if __name__ == "__main__":
    unittest.main()
