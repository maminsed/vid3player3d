"""Regression checks for shared correction and SMPL/pelvis coordinates.

Run from vid2player3d with:
    python -m unittest discover -s uhc/utils/tests -v
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from uhc.utils import convert_amass_isaac_correct_ground as ground

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR.parent))
from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree


class GroundCorrectionTests(unittest.TestCase):
    def make_sequence(self, **kwargs):
        # A nonzero offset on every axis catches accidental use of pelvis
        # coordinates as SMPL translation, including horizontal corrections.
        offset = np.array([0.12, -0.08, 0.25], dtype=np.float32)
        translations = np.tile([0.4, 0.6, 1.2], (30, 1)).astype(np.float32)
        entry = {
            'pose_aa': np.zeros((30, 72), dtype=np.float32),
            'trans_orig': translations,
            'beta': np.zeros(10, dtype=np.float32),
            'gender': 'neutral',
        }
        local_translations = torch.zeros(24, 3)
        local_translations[0] = torch.from_numpy(offset)
        tree = SkeletonTree(
            ['Pelvis'] + [f'joint_{i}' for i in range(23)],
            torch.tensor([-1] + [0] * 23), local_translations,
        )
        # Stand-in geometry avoids requiring licensed SMPL assets or MuJoCo.
        template = torch.zeros(6890, 3)
        template[:, 2] = torch.linspace(-0.9, 0.9, 6890)

        def mesh(pose, th_betas, th_trans):
            return template[None] + th_trans[:, None, :], None

        robot = SimpleNamespace(
            smpl_parser_n=SimpleNamespace(get_joints_verts=mesh),
            load_from_skeleton=Mock(), write_xml=Mock(), model_xml_path='unused.xml',
        )
        with patch.object(SkeletonTree, 'from_mjcf', return_value=tree):
            result = ground.correct_smpl_sequence(entry, robot, plot_results=False, **kwargs)
        np.testing.assert_array_equal(entry['trans_orig'], translations)
        return result, offset

    def test_mesh_and_skeleton_receive_same_displacement(self):
        sequence, offset = self.make_sequence()
        delta = sequence['corrected_root_trans'] - sequence['root_trans']
        expected_mesh = sequence['verts'].numpy() + delta[:, None, :]
        np.testing.assert_allclose(sequence['corrected_verts'], expected_mesh, atol=2e-7)
        np.testing.assert_allclose(
            sequence['corrected_trans'] + offset,
            sequence['corrected_root_trans'], atol=2e-7,
        )
        np.testing.assert_allclose(sequence['corrected_verts'][..., 2].min(dim=1)[0], 0, atol=2e-7)

    def test_horizontal_correction_uses_same_translation_convention(self):
        import pandas as pd

        context = {
            'enabled': True, 'frame_offset': 0, 'frame_window': 0,
            'frame_column': 'frame', 'point_column': 'person_point_bottom',
            'tp_data': pd.DataFrame({'frame': range(30), 'person_point_bottom': [[2., 4.]] * 30}),
            'net_center': np.zeros(2), 'xy_units': 'meters',
            'pixel_to_meters': 1., 'flip_y': False,
        }
        sequence, offset = self.make_sequence(xy_context=context)
        np.testing.assert_allclose(sequence['corrected_root_trans'][:, :2], np.tile([2., 4.], (30, 1)))
        delta = sequence['corrected_root_trans'] - sequence['root_trans']
        np.testing.assert_allclose(
            sequence['corrected_verts'], sequence['verts'].numpy() + delta[:, None, :], atol=5e-7,
        )

    def test_serialized_motion_and_render_mesh_stay_aligned_after_trimming(self):
        sequence, offset = self.make_sequence()
        motion, render = ground.build_motion_output(sequence, 'example', 0, 0, trim_frames=2)
        reconstructed = SkeletonMotion.from_dict(motion)
        np.testing.assert_allclose(motion['trans'] + offset, motion['root_trans'], atol=2e-7)
        np.testing.assert_allclose(reconstructed.root_translation, motion['root_trans'])
        np.testing.assert_allclose(render['verts'], sequence['corrected_verts'][2:-2])
        self.assertEqual(len(motion['pose_aa']), 26)
        self.assertAlmostEqual(motion['min_verts_h'], 0, places=6)

    def test_imports_ignore_cli_arguments_and_do_not_load_simulation(self):
        code = '''
import sys
sys.argv = ['caller', '--unrelated-option']
from uhc.utils import convert_amass_isaac_correct_ground as ground
from uhc.utils import plot_ground_contacts_simple as plot
assert plot.correct_smpl_sequence is ground.correct_smpl_sequence
assert plot.get_foot_vertices is ground.get_foot_vertices
assert plot.detect_ground_contacts_with_feet is ground.detect_ground_contacts_with_feet
assert 'mujoco_py' not in sys.modules
assert 'isaacgym' not in sys.modules
assert 'torch' not in sys.modules
'''
        env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR), MPLBACKEND='Agg')
        with tempfile.TemporaryDirectory() as directory:
            env['MPLCONFIGDIR'] = str(Path(directory) / 'matplotlib')
            subprocess.run([sys.executable, '-c', code], cwd=directory, env=env, check=True)
            self.assertFalse((Path(directory) / 'data').exists())

    def test_diagnostic_plot_uses_explicit_output_directory(self):
        sequence, _ = self.make_sequence()
        with tempfile.TemporaryDirectory() as directory:
            ground.correct_ground_height(
                sequence['verts'].numpy(), sequence['root_trans'],
                sequence_name='folder/example', out_dir=directory, show_plots=False,
            )
            self.assertTrue((Path(directory) / 'ground_correction_folder_example.png').is_file())


if __name__ == '__main__':
    unittest.main()
