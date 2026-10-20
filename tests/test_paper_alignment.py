"""Offline regression checks against manuscript Eq. (6) and serialization.

The loss tests execute the actual methods with scalar adapters; they do not
replace a real PyTorch/autograd integration test. Run: python -B -m unittest
discover -s tests -v
"""
import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

CODE = Path(__file__).resolve().parents[1] / "code"
spec = importlib.util.spec_from_file_location("alignment_protocol", CODE / "protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

class Scalar(float):
    def float(self): return self
    def detach(self): return self
    def item(self): return float(self)

def loss_method(name):
    tree = ast.parse((CODE / "loss_weighting.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LossWeighting")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": SimpleNamespace(exp=math.exp), "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(CODE / "loss_weighting.py"), "exec"), namespace)
    return namespace[name]

class PaperLossTests(unittest.TestCase):
    def test_adaptive_matches_equation_six_at_nonzero_log_variances(self):
        obj = SimpleNamespace(mode="adaptive", log_var_cls=Scalar(.3), log_var_lm=Scalar(-.2))
        expected = .5 * (math.exp(-.3) * 2 + .3) + .5 * (math.exp(.2) * 6 - .2)
        self.assertAlmostEqual(loss_method("forward")(obj, Scalar(2), Scalar(6), True), expected)

    def test_absent_rationale_masks_loss_and_its_regularizer(self):
        obj = SimpleNamespace(mode="adaptive", log_var_cls=Scalar(.3), log_var_lm=Scalar(-.2))
        self.assertAlmostEqual(loss_method("forward")(obj, Scalar(2), Scalar(float("nan")), False), .5 * (math.exp(-.3) * 2 + .3))

    def test_snapshot_reports_actual_equation_coefficients(self):
        obj = SimpleNamespace(mode="adaptive", log_var_cls=Scalar(0), log_var_lm=Scalar(0))
        snapshot = loss_method("snapshot")(obj)
        self.assertEqual(snapshot["classification_weight"], .5)
        self.assertEqual(snapshot["reasoning_weight"], .5)

    def test_fixed_control_ratios_are_unchanged(self):
        for ratio in (1, 3, 5):
            obj = SimpleNamespace(mode="fixed_" + str(ratio))
            self.assertEqual(loss_method("forward")(obj, Scalar(2), Scalar(6), True), 2 + ratio * 6)
            self.assertEqual(loss_method("forward")(obj, Scalar(2), Scalar(6), False), 2)

class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(zip(protocol.FEATURE_COLS, [12.5, 5.2, 600, 2.5, 3, 4, "2020-01-01 08:00:00", "commercial district", -2]))

    def tearDown(self):
        if hasattr(protocol, "configure_semantic_dictionary"):
            protocol.configure_semantic_dictionary(None)

    def test_cost_fields_have_usd_units(self):
        text = protocol.feature_text(self.row)
        self.assertIn("12.5 USD", text)
        self.assertIn("2.5 USD", text)

    def test_readable_categories_are_preserved(self):
        self.assertIn("commercial district", protocol.feature_text(self.row))
        self.assertIn("2020-01-01 08:00:00", protocol.feature_text(self.row))

    def test_encoded_categories_require_real_dictionary(self):
        self.row["dropoff_poi_category"] = 3
        with self.assertRaisesRegex(ValueError, "semantic_dictionary"):
            protocol.feature_text(self.row)

    def test_dictionary_decodes_both_categorical_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "semantic_dictionary.json"
            path.write_text(json.dumps({"trip_start_timestamp": {"2": "2020-01-01 08:00:00"}, "dropoff_poi_category": {"3": "commercial district"}}), encoding="utf-8")
            metadata = protocol.configure_semantic_dictionary(folder)
            self.row.update(trip_start_timestamp=2, dropoff_poi_category=3)
            text = protocol.feature_text(self.row)
            self.assertIn("commercial district", text)
            self.assertIn("2020-01-01 08:00:00", text)
            self.assertEqual(metadata["semantic_dictionary_sha256"], protocol.sha256(path))

    def test_integer_code_read_as_float_uses_same_dictionary_key(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "semantic_dictionary.json").write_text(json.dumps({"dropoff_poi_category": {"3": "commercial district"}}), encoding="utf-8")
            protocol.configure_semantic_dictionary(folder)
            self.row["dropoff_poi_category"] = 3.0
            self.assertIn("commercial district", protocol.feature_text(self.row))

    def test_unmapped_numeric_category_is_not_rendered_as_semantic_text(self):
        self.row["dropoff_poi_category"] = "3.5"
        with self.assertRaisesRegex(ValueError, "semantic_dictionary"):
            protocol.feature_text(self.row)

    def test_labels_and_shap_are_not_serialized(self):
        self.row.update(shared_trip_authorized="SECRET_LABEL", sample_id="SECRET_ID", shap="SECRET_SHAP")
        text = protocol.feature_text(self.row)
        for secret in ("SECRET_LABEL", "SECRET_ID", "SECRET_SHAP"):
            self.assertNotIn(secret, text)

if __name__ == "__main__": unittest.main()
