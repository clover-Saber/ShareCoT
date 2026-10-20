"""API protocol regression tests; the transport is always offline."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import api_suite as api

class OfflineClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"reason":"test","pred":0}'))], id="offline", model=kwargs["model"], usage=None)

class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        self.client = OfflineClient()
        self.transport = patch.object(api, "client_for_suite", return_value=(self.client, "offline-placeholder"))
        self.transport.start()
        self.features = {key: "10" for key in api.FEATURE_COLS}
        self.features.update(trip_start_timestamp="2020-01-01 08:00:00", dropoff_poi_category="office")
        self.write_csv("test", ["test-A"], [0])
    def tearDown(self):
        self.transport.stop()
        self.temp.cleanup()
    def write_csv(self, name, ids, labels):
        import pandas as pd
        pd.DataFrame([dict(self.features, sample_id=sid, **{api.TARGET: label}) for sid, label in zip(ids, labels)]).to_csv(self.root / "data" / (name + ".csv"), index=False)
    def train_reasons(self, ids=("0001", "0002", "0003", "0004")):
        self.write_csv("train", ids, [0, 0, 1, 1])
        (self.root / "data" / "rationales.jsonl").write_text("".join(json.dumps({"sample_id": sid, "accepted": True, "reason": "A grounded explanation."}) + "\n" for sid in ids), encoding="utf-8")
    def run_api(self, shots):
        with contextlib.redirect_stdout(io.StringIO()):
            return api.run(self.root, workers=1, models=["gpt-5.4"], shots_filter=shots)
    def test_zero_shot_requires_no_training_or_rationale_files(self):
        self.assertEqual(self.run_api(0), 0)
        self.assertEqual(len(self.client.calls[0]["messages"]), 2)
    def test_four_shot_preserves_numeric_string_sample_ids(self):
        self.train_reasons()
        self.assertEqual(self.run_api(4), 0)
        messages = self.client.calls[0]["messages"]
        self.assertEqual([json.loads(m["content"])["pred"] for m in messages if m["role"] == "assistant"], [0, 0, 1, 1])
    def test_endpoint_change_rejects_stale_cache(self):
        self.train_reasons(("a", "b", "c", "d"))
        with patch.dict(os.environ, {"API_BASE_URL": "https://endpoint-a.invalid/v1"}): self.run_api(0)
        with patch.dict(os.environ, {"API_BASE_URL": "https://endpoint-b.invalid/v1"}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.run_api(0)
        self.assertEqual(len(self.client.calls), 1)
    def test_dictionary_is_loaded_before_prompt_construction(self):
        self.train_reasons(("a", "b", "c", "d"))
        self.features["dropoff_poi_category"] = "3"
        self.write_csv("test", ["test-A"], [0])
        (self.root / "data" / "semantic_dictionary.json").write_text(json.dumps({"dropoff_poi_category": {"3": "commercial district"}}), encoding="utf-8")
        self.assertEqual(self.run_api(0), 0)
        self.assertIn("commercial district", self.client.calls[0]["messages"][-1]["content"])

if __name__ == "__main__": unittest.main()
