"""CPU contracts for fixed-asset fine-tuning (no simulator required)."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from embodied_pose.utils.imitation_asset import read_fixed_asset, validate_motion_asset

ASSET = PROJECT / 'vid2player/data/assets/smpl_mesh_humanoid_djokovic.xml'


class FixedAssetTests(unittest.TestCase):
    def setUp(self):
        self.names, self.parents, self.offsets, self.rest = read_fixed_asset(ASSET)

    def library(self):
        # Cleaned MotionLib-style state, with nonidentity global rotations.
        q = torch.tensor([0., 0., np.sin(.3), np.cos(.3)], dtype=torch.float32)
        v = torch.tensor(self.rest)
        uv = torch.cross(q[:3].expand_as(v), v, dim=-1)
        pos = v + 2 * (q[3] * uv + torch.cross(q[:3].expand_as(v), uv, dim=-1))
        return SimpleNamespace(
            _motion_bodies=torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            _motion_body_scales=torch.ones(2), length_starts=torch.tensor([0, 3]),
            _motion_num_frames=torch.tensor([3, 3]),
            gts=pos.expand(6, -1, -1).clone(), grs=q.expand(6, 24, -1).clone())

    def validate(self, lib):
        validate_motion_asset(lib, self.names, self.parents, self.offsets, [1, 0, 0], 1.)

    def test_cleaned_library_geometry_and_quaternion_sign(self):
        lib = self.library()
        lib.grs[1::2] *= -1
        self.validate(lib)
        lib.gts[4, 20, 0] += .01
        with self.assertRaisesRegex(ValueError, 'skeleton'):
            self.validate(lib)

    def test_conditioning_and_scale_mismatch(self):
        lib = self.library()
        lib._motion_bodies[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'conditioning'):
            self.validate(lib)
        lib = self.library()
        lib._motion_body_scales[1] = 1.1
        with self.assertRaisesRegex(ValueError, 'scale'):
            self.validate(lib)

    def test_optional_racket_and_invalid_attachment(self):
        self.assertEqual(self.names[-1], 'Racket')
        self.assertEqual(self.parents[-1], self.names.index('R_Wrist'))
        doc = ET.parse(ASSET)
        wrist = doc.find(".//body[@name='R_Wrist']")
        racket = wrist.find("body[@name='Racket']")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'asset.xml'
            ET.SubElement(racket, 'joint', name='racket_x', type='hinge')
            doc.write(path)
            with self.assertRaises(ValueError):
                read_fixed_asset(path)
            wrist.remove(racket)
            doc.write(path)
            self.assertEqual(len(read_fixed_asset(path)[0]), 24)

    def test_reject_changed_humanoid_hierarchy(self):
        doc = ET.parse(ASSET)
        wrist = doc.find(".//body[@name='R_Wrist']")
        hand = wrist.find("body[@name='R_Hand']")
        wrist.remove(hand)
        doc.find(".//body[@name='R_Elbow']").append(hand)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'asset.xml'
            doc.write(path)
            with self.assertRaises(ValueError):
                read_fixed_asset(path)

    def test_training_and_deployment_dynamics_match(self):
        ft = yaml.safe_load((PROJECT/'embodied_pose/cfg/tennis_im_training_2_ft.yaml').read_text())
        high = yaml.safe_load((PROJECT/'vid2player/cfg/controller/usopen_train_stage_1/high_level_6.yaml').read_text())['env']
        self.assertEqual(ft['env']['racket_dynamics'], high['vid2player']['racket_dynamics'])
        self.assertEqual(ft['env']['tennis_contacts'], high['vid2player']['tennis_contacts'])
        for key in ('dt', 'substeps'):
            self.assertEqual(ft['sim'][key], high['physics'][key])
        for key in ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                    'contact_offset', 'rest_offset', 'bounce_threshold_velocity',
                    'max_depenetration_velocity', 'friction_offset_threshold', 'friction_correlation_distance'):
            self.assertEqual(ft['sim']['physx'][key], high['physics'][key])
        for key in ('controlFrequencyInv', 'residual_force_scale'):
            self.assertEqual(ft['env'][key], high['physics'][key])


if __name__ == '__main__':
    unittest.main()
