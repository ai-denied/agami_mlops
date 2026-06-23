import json
import os
import tempfile
import unittest

from context_emotion.scripts import package_emotion_model
from context_emotion.tests.fixtures import VERSION, write_valid_candidate_inputs


class TestPackageContract(unittest.TestCase):
    def test_valid_inputs_produce_a_candidate_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_valid_candidate_inputs(os.path.join(tmp, "run_output"))
            output_dir = os.path.join(tmp, "candidates", VERSION)

            final_dir = package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)

            self.assertEqual(final_dir, output_dir)
            for name in ("model.onnx", "metadata.json", "label_schema.json",
                         "preprocessing_config.json", "evaluation_result.json", "manifest.json"):
                self.assertTrue(os.path.isfile(os.path.join(final_dir, name)), f"missing {name}")

            with open(os.path.join(final_dir, "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["version"], VERSION)
            self.assertEqual(set(manifest["files"]), {
                "model.onnx", "metadata.json", "label_schema.json",
                "preprocessing_config.json", "evaluation_result.json",
            })

    def test_missing_input_file_blocks_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_valid_candidate_inputs(os.path.join(tmp, "run_output"))
            inputs["label_schema"] = os.path.join(tmp, "run_output", "does_not_exist.json")
            output_dir = os.path.join(tmp, "candidates", VERSION)

            with self.assertRaises(FileNotFoundError):
                package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)
            self.assertFalse(os.path.isdir(output_dir), "candidate dir must not exist after a failed package()")

    def test_label_schema_mismatch_blocks_candidate_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_output")
            inputs = write_valid_candidate_inputs(run_dir)

            # corrupt label_schema.json: wrong class order
            broken_schema_path = os.path.join(run_dir, "label_schema.json")
            with open(broken_schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            schema["emotion_classes"] = list(reversed(schema["emotion_classes"]))
            with open(broken_schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f)

            output_dir = os.path.join(tmp, "candidates", VERSION)
            with self.assertRaises(ValueError) as ctx:
                package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)
            self.assertIn("emotion_classes", str(ctx.exception))
            self.assertFalse(os.path.isdir(output_dir), "후보 생성 자체가 막혀야 함 (요구사항)")

    def test_onnx_hash_mismatch_blocks_candidate_creation(self):
        """evaluation_result.json must describe THIS exact onnx file - if the
        onnx changes after evaluate_candidate.py ran (or someone copy-pastes
        an old evaluation_result.json next to a new onnx), packaging must
        refuse rather than silently shipping untested numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_output")
            inputs = write_valid_candidate_inputs(run_dir)

            # swap in a different onnx after evaluation_result.json was written
            with open(inputs["onnx"], "wb") as f:
                f.write(b"a completely different (and untested) onnx file")

            output_dir = os.path.join(tmp, "candidates", VERSION)
            with self.assertRaises(ValueError) as ctx:
                package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)
            self.assertIn("onnx_sha256", str(ctx.exception))
            self.assertFalse(os.path.isdir(output_dir), "후보 생성 자체가 막혀야 함")

    def test_version_mismatch_blocks_candidate_creation(self):
        """metadata.json's own 'version' field must agree with the
        candidates/{version}/ directory name / --version flag - otherwise
        current/metadata.json after promotion could silently disagree with
        which candidate it actually came from."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_output")
            inputs = write_valid_candidate_inputs(run_dir, version="v_built_as")

            output_dir = os.path.join(tmp, "candidates", "v_shipped_as")
            with self.assertRaises(ValueError) as ctx:
                package_emotion_model.package(**inputs, version="v_shipped_as", output_dir=output_dir)
            self.assertIn("version", str(ctx.exception))
            self.assertFalse(os.path.isdir(output_dir), "후보 생성 자체가 막혀야 함")

    def test_cannot_package_directly_into_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_valid_candidate_inputs(os.path.join(tmp, "run_output"))
            from context_emotion.deployment import model_store
            with self.assertRaises(ValueError):
                package_emotion_model.package(**inputs, version=VERSION, output_dir=model_store.CURRENT_DIR)

    def test_duplicate_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_valid_candidate_inputs(os.path.join(tmp, "run_output"))
            output_dir = os.path.join(tmp, "candidates", VERSION)
            package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)

            with self.assertRaises(ValueError):
                package_emotion_model.package(**inputs, version=VERSION, output_dir=output_dir)


if __name__ == "__main__":
    unittest.main()
