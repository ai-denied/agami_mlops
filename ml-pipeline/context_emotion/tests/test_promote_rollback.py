import json
import os
import tempfile
import unittest

from context_emotion.deployment import model_store, promote_model, rollback_model
from context_emotion.scripts import package_emotion_model
from context_emotion.tests.fixtures import write_valid_candidate_inputs


class TestPromoteRollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store_root = os.path.join(self.tmp.name, "model-store", "context_emotion")
        _, self.candidates_dir, self.current_dir, self.archive_dir = model_store.resolve_store_paths(self.store_root)

    def tearDown(self):
        self.tmp.cleanup()

    def _package(self, version: str):
        inputs = write_valid_candidate_inputs(os.path.join(self.tmp.name, f"run_{version}"), version=version)
        output_dir = os.path.join(self.candidates_dir, version)
        return package_emotion_model.package(**inputs, version=version, output_dir=output_dir)

    def test_first_promotion_with_no_prior_current(self):
        self._package("v1")
        result = promote_model.promote(version="v1", skip_validate=True, store_root_override=self.store_root)

        self.assertTrue(result["promoted"])
        self.assertTrue(os.path.isfile(os.path.join(self.current_dir, "model.onnx")))
        with open(os.path.join(self.current_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["version"], "v1")
        self.assertIn("promoted_at", meta)

    def test_promote_then_rollback_restores_previous_version(self):
        self._package("v1")
        promote_model.promote(version="v1", skip_validate=True, store_root_override=self.store_root)

        self._package("v2")
        promote_model.promote(version="v2", skip_validate=True, store_root_override=self.store_root)

        with open(os.path.join(self.current_dir, "metadata.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["version"], "v2")

        archives = rollback_model.list_archives(self.archive_dir)
        self.assertTrue(any("v1" in a for a in archives), f"expected a v1 backup in {archives}")

        rollback_model.rollback(store_root_override=self.store_root)

        with open(os.path.join(self.current_dir, "metadata.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["version"], "v1")

    def test_promote_missing_candidate_raises_without_touching_current(self):
        with self.assertRaises(FileNotFoundError):
            promote_model.promote(version="does_not_exist", skip_validate=True, store_root_override=self.store_root)
        self.assertFalse(os.path.isdir(self.current_dir))

    def test_dry_run_promote_does_not_change_current(self):
        self._package("v1")
        promote_model.promote(version="v1", dry=True, skip_validate=True, store_root_override=self.store_root)
        self.assertFalse(os.path.isdir(self.current_dir), "dry-run must not create/modify current/")

    def test_rollback_with_empty_archive_raises(self):
        with self.assertRaises(FileNotFoundError):
            rollback_model.rollback(store_root_override=self.store_root)


if __name__ == "__main__":
    unittest.main()
