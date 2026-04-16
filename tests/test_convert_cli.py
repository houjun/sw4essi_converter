import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The CLI parser test does not execute MPI or compressed HDF5 code paths.
# Stub these imports so CI does not need native MPI tooling just to import convert.py.
mpi4py_stub = types.ModuleType("mpi4py")
mpi4py_stub.MPI = types.SimpleNamespace()
sys.modules.setdefault("mpi4py", mpi4py_stub)
sys.modules.setdefault("hdf5plugin", types.ModuleType("hdf5plugin"))

import convert


class ConvertCliParserTests(unittest.TestCase):
    def setUp(self):
        self.sample_ssi = REPO_ROOT / "tests" / "data" / "small.ssi"
        self.assertTrue(self.sample_ssi.exists(), f"Missing test fixture: {self.sample_ssi}")
        self.parser = convert.build_arg_parser()

    def test_primary_ssi_flag_uses_repo_fixture(self):
        args = self.parser.parse_args(["--ssi", str(self.sample_ssi)])
        self.assertEqual(args.ssi, str(self.sample_ssi))

    def test_legacy_essi_flag_maps_to_ssi_destination(self):
        args = self.parser.parse_args(["--essi", str(self.sample_ssi)])
        self.assertEqual(args.ssi, str(self.sample_ssi))


if __name__ == "__main__":
    unittest.main()
