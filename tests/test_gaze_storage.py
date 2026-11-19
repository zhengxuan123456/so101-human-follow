import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from so101_human_follow.gaze_storage import atomic_write_gaze, gaze_state_path, read_gaze


class GazeStorageTest(unittest.TestCase):
    def test_project_namespaces_are_stable_and_do_not_share_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project-a"
            first = gaze_state_path(project, data_root=root / "data")
            normalized = gaze_state_path(project / "src" / "..", data_root=root / "data")
            other = gaze_state_path(root / "project-b", data_root=root / "data")
            self.assertEqual(first, normalized)
            self.assertNotEqual(first, other)
            self.assertEqual(len(first.parent.name), 16)
            self.assertEqual(first.name, "gaze_calibration.json")
            self.assertEqual(first.parent.parent.name, "so101-human-follow")
            self.assertFalse(first.parent.exists())  # resolving paths creates no files

    def test_windows_uses_local_app_data_and_fallback_uses_home_share(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local")}), patch(
                "so101_human_follow.gaze_storage.Path.home", return_value=root / "home",
            ):
                location = gaze_state_path(root / "project")
                expected_root = root / "local" if os.name == "nt" else root / "home" / ".local" / "share"
                self.assertEqual(location.parent.parent.parent, expected_root)
            with patch.dict(os.environ, {"LOCALAPPDATA": ""}), patch(
                "so101_human_follow.gaze_storage.Path.home", return_value=root / "home",
            ):
                fallback = gaze_state_path(root / "project")
                self.assertEqual(fallback.parent.parent.parent, root / "home" / ".local" / "share")

    def test_atomic_write_creates_parent_replaces_old_state_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "private" / "nested" / "gaze.json"
            self.assertIsNone(read_gaze(location))
            payload = {"version": 1, "jacobian": [[-0.02, 0.01], [0.01, 0.02]], "description": "注视方向"}
            atomic_write_gaze(location, payload)
            self.assertEqual(read_gaze(location), payload)
            atomic_write_gaze(location, {"version": 1, "invalidated": True})
            self.assertEqual(read_gaze(location), {"version": 1, "invalidated": True})
            self.assertEqual(list(location.parent.iterdir()), [location])

    def test_replace_failure_preserves_old_state_and_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "gaze.json"
            atomic_write_gaze(location, {"old": True})
            error = PermissionError("replacement denied")
            with patch("so101_human_follow.gaze_storage.os.replace", side_effect=error):
                with self.assertRaises(PermissionError) as raised:
                    atomic_write_gaze(location, {"new": True})
            self.assertIs(raised.exception, error)
            self.assertEqual(read_gaze(location), {"old": True})
            self.assertEqual(list(location.parent.iterdir()), [location])

    def test_cleanup_failure_is_logged_without_masking_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "gaze.json"
            error = PermissionError("original replacement denied")
            with patch("so101_human_follow.gaze_storage.os.replace", side_effect=error), patch(
                "so101_human_follow.gaze_storage.Path.unlink", side_effect=PermissionError("cleanup denied"),
            ), self.assertLogs("so101_human_follow.gaze_storage", level="WARNING") as messages:
                with self.assertRaises(PermissionError) as raised:
                    atomic_write_gaze(location, {"new": True})
            self.assertIs(raised.exception, error)
            self.assertIn("Could not remove gaze temporary file", messages.output[0])
            for own_temporary in Path(directory).glob(".gaze-*.tmp"):
                own_temporary.unlink()  # only this test's own files, after unpatching

    def test_nonfinite_or_nonobject_payload_does_not_touch_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "gaze.json"
            atomic_write_gaze(location, {"valid": True})
            for invalid in ({"value": float("nan")}, {"value": float("inf")}, []):
                with self.assertRaises(ValueError):
                    atomic_write_gaze(location, invalid)
                self.assertEqual(read_gaze(location), {"valid": True})
            self.assertEqual(list(location.parent.iterdir()), [location])

    def test_corrupt_nonfinite_and_nonobject_read_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory) / "gaze.json"
            for contents in ('{"version":', '[]', 'null', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
                location.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_gaze(location)
            location.write_bytes(b"\xff\xfe\x00")
            with self.assertRaises(ValueError):
                read_gaze(location)

    def test_read_permission_errors_remain_io_errors(self):
        with patch("so101_human_follow.gaze_storage.Path.read_text", side_effect=PermissionError("read denied")):
            with self.assertRaises(PermissionError):
                read_gaze(Path("explicit-test-location.json"))


if __name__ == "__main__":
    unittest.main()
