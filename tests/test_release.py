# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import json
import subprocess
import sys
import unittest

from _support import ROOT


class ReleaseTests(unittest.TestCase):
    def test_release_check_passes(self):
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "release_check.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_no_rust_anywhere(self):
        rust = [p for p in ROOT.rglob("*") if p.suffix == ".rs" or p.name.startswith("Cargo.")]
        self.assertEqual(rust, [])

    def test_manifest_matches_the_package(self):
        manifest = json.loads((ROOT / "audit-manifest.json").read_text())
        self.assertEqual(manifest["package"], "omnievolve")
        self.assertEqual(manifest["license"], "Apache-2.0")
        self.assertIs(manifest["rust_included"], False)
        self.assertTrue(manifest["public_repository"].endswith("/omnievolve"))
        for name in manifest["package_files"]:
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
