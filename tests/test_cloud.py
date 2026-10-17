# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _support import EXAMPLES

from omnievolve.cloud import CloudClient, CloudError, build_payload, summarize_payload
from omnievolve.config import load_config


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="omnievolve-cloud-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.project = self.dir / "proj"
        shutil.copytree(EXAMPLES / "prompt_evolution", self.project)
        (self.project / "deploy.pem").write_text("not a real key\n")
        (self.project / ".omnievolve").mkdir()
        (self.project / ".omnievolve" / "ledger.db").write_text("x")

    def test_payload_shape_and_exclusions(self):
        body = build_payload(load_config(self.project / "omnievolve.json"), local_run_id="run-1")
        spec = body["spec"]
        self.assertEqual(body["generations"], 10)
        self.assertEqual(body["candidates_per_generation"], 16)
        self.assertEqual(body["run_id"], "run-1")
        self.assertEqual(spec["target"]["path"], "system_prompt.txt")
        self.assertEqual(spec["objectives"], [{"name": "score", "direction": "maximize"}])
        self.assertEqual(spec["features"]["selection"], "pareto_frontier")
        self.assertIn("mutation_pool.txt", spec["files"])
        self.assertNotIn("deploy.pem", spec["files"])
        self.assertIn("deploy.pem", spec["skipped_files"])
        self.assertFalse(any(name.startswith(".omnievolve") for name in spec["files"]))
        summary = summarize_payload(body)
        self.assertIsInstance(summary["spec"]["files"]["evaluate.py"], int)
        self.assertNotIn("source", summary["spec"]["target"])

    def test_client_submits_with_bearer_token(self):
        sent = []

        def transport(method, url, headers, payload, timeout):
            sent.append((method, url, headers, payload))
            if method == "POST":
                return {"run": {"id": "evo_1", "status": "queued"}}
            return {"run": {"id": "evo_1", "status": "completed"}}

        client = CloudClient("https://cloud.example.com/", "k-123", transport=transport)
        self.assertEqual(client.submit({"generations": 1})["run"]["id"], "evo_1")
        final = client.wait("evo_1", sleep=lambda s: None)
        self.assertEqual(final["status"], "completed")
        method, url, headers, _ = sent[0]
        self.assertEqual((method, url), ("POST", "https://cloud.example.com/api/evolve/runs"))
        self.assertEqual(headers["authorization"], "Bearer k-123")
        self.assertEqual(sent[1][1], "https://cloud.example.com/api/evolve/runs/evo_1")

    def test_client_requires_https_and_credentials(self):
        with mock.patch.dict("os.environ", {"OMNISI_CLOUD_URL": "", "OMNISI_API_KEY": ""}):
            with self.assertRaisesRegex(CloudError, "OMNISI_CLOUD_URL"):
                CloudClient()
            with self.assertRaisesRegex(CloudError, "OMNISI_API_KEY"):
                CloudClient("https://cloud.example.com")
        with self.assertRaisesRegex(CloudError, "https"):
            CloudClient("http://cloud.example.com", "k")
        CloudClient("http://localhost:8080", "k")  # local development is allowed


if __name__ == "__main__":
    unittest.main()
