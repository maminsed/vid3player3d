"""CPU regressions for leg-label repair and report-only HMR4D diagnostics."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from GVHMR.hmr4d.utils.motion_diagnostics import (
    detect_short_lr_swaps, repair_short_lr_swaps, save_lr_swap_report,
    final_motion_metrics, rotation_steps, save_hmr4d_spike_plot,
)
from GVHMR.hmr4d.utils.vis.cv2_utils import draw_coco17_skeleton


class MotionDiagnosticsTests(unittest.TestCase):
    def motion(self, length=30):
        kp = np.zeros((length, 17, 3), dtype=np.float32)
        kp[..., 2] = 0.9
        for j in range(5, 17):
            kp[:, j, :2] = [30 if j % 2 else 70, j * 3]
        kp[..., 0] += np.arange(length)[:, None] * 8  # Rapid camera/player translation.
        boxes = np.tile([50, 50, 100], (length, 1)).astype(np.float32)
        boxes[:, 0] += np.arange(length) * 8
        return kp, boxes

    def swap(self, kp, start, stop, joints):
        for left, right in joints:
            kp[start:stop, [left, right]] = kp[start:stop, [right, left]].copy()

    def test_short_leg_swaps_without_modifying_input(self):
        for size in (1, 2, 3):
            with self.subTest(size=size):
                kp, boxes = self.motion()
                self.swap(kp, 12, 12 + size, [[11, 12], [13, 14], [15, 16]])
                before = kp.copy()
                events = detect_short_lr_swaps(kp, boxes)
                self.assertEqual([(e["limb"], e["start_frame"], e["end_frame"]) for e in events],
                                 [("legs", 12, 11 + size)])
                np.testing.assert_array_equal(kp, before)

    def test_arm_swaps_are_excluded_from_repair(self):
        kp, boxes = self.motion()
        self.swap(kp, 12, 13, [[5, 6], [7, 8], [9, 10]])
        repaired, events = repair_short_lr_swaps(kp, boxes)
        self.assertEqual(events, [])
        np.testing.assert_array_equal(kp, repaired)

    def test_repair_restores_labels_and_scores_for_numpy_and_torch(self):
        for size in (1, 2, 3):
            for as_tensor in (False, True):
                with self.subTest(size=size, as_tensor=as_tensor):
                    clean, boxes = self.motion()
                    clean[:, 11::2, 2] = 0.65
                    clean[:, 12::2, 2] = 0.85
                    swapped = clean.copy()
                    self.swap(swapped, 12, 12 + size, [[11, 12], [13, 14], [15, 16]])
                    original = swapped.copy()
                    source = torch.from_numpy(swapped) if as_tensor else swapped
                    repaired, events = repair_short_lr_swaps(source, boxes)
                    np.testing.assert_array_equal(repaired, clean)
                    np.testing.assert_array_equal(source, original)
                    self.assertEqual(isinstance(repaired, torch.Tensor), as_tensor)
                    self.assertTrue(events[0]["repaired"])
                    # Rechecking repaired observations must not undo the correction.
                    again, repeated = repair_short_lr_swaps(repaired, boxes)
                    self.assertEqual(repeated, [])
                    np.testing.assert_array_equal(again, repaired)

    def test_only_supported_pairs_are_repaired_and_audited(self):
        clean, boxes = self.motion()
        swapped = clean.copy()
        self.swap(swapped, 12, 13, [[13, 14], [15, 16]])
        repaired, events = repair_short_lr_swaps(swapped, boxes)
        np.testing.assert_array_equal(repaired, clean)
        self.assertEqual(events[0]["pairs"], [[13, 14], [15, 16]])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "swaps.json"
            save_lr_swap_report(events, path)
            report = json.loads(path.read_text())
            self.assertFalse(report["diagnostic_only"])
            self.assertTrue(report["events"][0]["repaired"])

    def test_fast_turn_and_fast_translation_are_not_swaps(self):
        kp, boxes = self.motion()
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])
        # A continuous 180-degree turn over six frames; anatomical identities stay fixed.
        angle = np.clip((np.arange(len(kp)) - 10) / 6, 0, 1) * np.pi
        for j in range(5, 17):
            kp[:, j, 0] = boxes[:, 0] + (-20 if j % 2 else 20) * np.cos(angle)
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])

    def test_long_or_edge_swaps_are_not_flagged(self):
        for start, stop in [(12, 16), (12, 30), (0, 2)]:
            kp, boxes = self.motion()
            self.swap(kp, start, stop, [[11, 12], [13, 14], [15, 16]])
            self.assertEqual(detect_short_lr_swaps(kp, boxes), [])

    def test_single_pair_low_confidence_and_missing_evidence_are_ignored(self):
        kp, boxes = self.motion()
        self.swap(kp, 12, 13, [[15, 16]])
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])
        self.swap(kp, 12, 13, [[11, 12], [13, 14]])
        kp[12, 11:, 2] = 0.4
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])
        kp[12, 11:, :2] = np.nan
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])

    def test_rotation_representation_wrap_does_not_create_spike(self):
        angles = np.zeros((2, 3))
        angles[:, 1] = np.deg2rad([179, -179])
        steps = rotation_steps(angles)
        self.assertTrue(np.isnan(steps[0]))
        self.assertAlmostEqual(steps[1], 2, places=5)

    def prediction(self, length=31):
        params = {"global_orient": torch.zeros(length, 3), "body_pose": torch.zeros(length, 63),
                  "transl": torch.zeros(length, 3)}
        params["global_orient"][15, 1] = np.pi / 2
        params["body_pose"][15, 0] = np.pi / 3
        return {"smpl_params_incam": params, "smpl_params_global": params,
                # Deliberately contradictory raw data: metrics must use FINAL parameters.
                "net_outputs": {"decode_dict": {"body_pose": torch.zeros(length, 63)}}}

    def test_plot_final_parameters_csv_alignment_and_no_mutation(self):
        pred = self.prediction()
        before = {k: v.clone() for k, v in pred["smpl_params_incam"].items()}
        metrics = final_motion_metrics(pred)
        self.assertAlmostEqual(metrics["legs_max_deg"][15], 60, places=4)
        with TemporaryDirectory() as directory:
            plot, csv_path = Path(directory) / "spikes.png", Path(directory) / "spikes.csv"
            flags = save_hmr4d_spike_plot(pred, plot, csv_path)
            self.assertEqual(flags["root_incam_deg"], [15, 16])
            self.assertGreater(plot.stat().st_size, 1000)
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 31)
            self.assertEqual(rows[15]["frame"], "15")
            self.assertEqual(float(rows[15]["seconds"]), 0.5)
            self.assertEqual(rows[15]["root_incam_deg_spike"], "1")
        for key, value in before.items():
            self.assertTrue(torch.equal(value, pred["smpl_params_incam"][key]))

    def test_single_frame_has_no_motion(self):
        pred = self.prediction()
        pred = {space: {key: value[:1] for key, value in pred[space].items()}
                for space in ["smpl_params_incam", "smpl_params_global"]}
        self.assertTrue(all(np.isnan(values).all() for values in final_motion_metrics(pred).values()))
        kp, boxes = self.motion(length=1)
        self.assertEqual(detect_short_lr_swaps(kp, boxes), [])

    def test_rgb_sides_match_for_hands_and_feet(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        kp = np.full((17, 3), np.nan)
        for j, xy in [(9, (20, 20)), (10, (180, 20)), (15, (20, 180)), (16, (180, 180))]:
            kp[j] = [*xy, 0.9]
        overlay = draw_coco17_skeleton(frame, kp)
        np.testing.assert_array_equal(overlay[20, 20], overlay[180, 20])
        np.testing.assert_array_equal(overlay[20, 180], overlay[180, 180])
        self.assertFalse(np.array_equal(overlay[20, 20], overlay[20, 180]))
        self.assertFalse(frame.any())


if __name__ == "__main__":
    unittest.main()
