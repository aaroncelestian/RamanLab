import pathlib
import os
import pickle
import tempfile
import unittest

from pkl_utils import safe_pickle_load


class CrossPlatformPickleTests(unittest.TestCase):
    def test_safe_pickle_load_handles_windows_path_on_non_windows(self):
        payload = b"cpathlib\nWindowsPath\np0\n(VC:/data/map/source.txt\np1\ntp2\nRp3\n."

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_file = pathlib.Path(tmp_dir) / "windows_path.pkl"
            pkl_file.write_bytes(payload)

            loaded = safe_pickle_load(pkl_file)

        self.assertEqual(loaded, pathlib.PureWindowsPath("C:/data/map/source.txt"))

    @unittest.skipIf(os.name == "nt", "PosixPath is only native on POSIX systems")
    def test_safe_pickle_load_preserves_native_posix_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_file = pathlib.Path(tmp_dir) / "posix_path.pkl"
            expected = pathlib.Path("/tmp/map/source.txt")
            pkl_file.write_bytes(pickle.dumps(expected))

            loaded = safe_pickle_load(pkl_file)

        self.assertIs(type(loaded), pathlib.PosixPath)
        self.assertEqual(loaded, expected)


if __name__ == "__main__":
    unittest.main()
