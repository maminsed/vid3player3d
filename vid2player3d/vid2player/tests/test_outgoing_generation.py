"""Outgoing row layout, interpolation and complete streamed batch coverage."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vid2player.utils.ball_outgoing_generation import decode_rows, interpolate_tables, generate_tables


PARAMS = SimpleNamespace(VEL_X_RANGE=(10, 12, 1), VEL_Y_RANGE=(-1, 2, 1),
                         VSPIN_RANGE=(-2, 3, 2), TRAJ_X_RANGE=(0, 25, .5), TRAJ_Y_RANGE=(0, 3, .1))


class TestOutgoingGeneration(unittest.TestCase):
    def test_runtime_row_order(self):
        expected = np.array([(v, z, w) for v in (10, 11) for z in (-1, 0, 1) for w in (-2, 0, 2)], dtype=np.float32)
        np.testing.assert_array_equal(decode_rows(np.arange(18), PARAMS), expected)
        with self.assertRaises(ValueError):
            decode_rows([18], PARAMS)

    def test_analytic_ballistic_interpolation_and_descending_zero(self):
        t = torch.arange(0, 3, .001, dtype=torch.float64)
        flight = torch.stack((10*t, 3*t-.5*9.81*t*t), dim=-1)[None]
        x, y = interpolate_tables(flight, .001, PARAMS)
        tx = np.arange(0, 25, .5) / 10
        np.testing.assert_allclose(x[0], 3*tx-.5*9.81*tx**2, atol=2e-6)
        times = (3+np.sqrt(9+2*9.81*np.arange(0,3,.1))) / 9.81
        np.testing.assert_allclose(y[0,:,0], times*10, atol=1e-5)
        np.testing.assert_allclose(y[0,:,1], times, atol=1e-6)
        self.assertGreater(y[0,0,1], .6)  # return to height zero, not launch at t=0

    def test_uncovered_flight_fails_instead_of_extrapolating(self):
        flight = torch.tensor([[[0.,0.], [1.,-.01], [2.,-.05]]])
        with self.assertRaises(ValueError):
            interpolate_tables(flight, .1, PARAMS)

    def test_distance_after_all_supported_landings_is_invalid_for_net(self):
        t = torch.arange(0, 1.5, .001, dtype=torch.float64)
        flight = torch.stack((10*t, 3*t-.5*9.81*t*t), dim=-1)[None]
        x, y = interpolate_tables(flight, .001, PARAMS, allow_landed_tail=True)
        self.assertTrue((x[0,30:] == -1e6).all())
        self.assertTrue(np.isfinite(y).all())
        with self.assertRaises(ValueError):
            interpolate_tables(flight[:,:10], .001, PARAMS, allow_landed_tail=True)

    def test_tail_padding_and_atomic_publication(self):
        calls = []
        def fake_sim(gen, launches, params, seconds):
            calls.extend(launches.tolist())
            return np.repeat(launches[:, :1], 50, axis=1), np.repeat(launches[:, None, 1:], 30, axis=1)
        gen = SimpleNamespace(num_env=5, physics_dt=1/60)
        with tempfile.TemporaryDirectory() as tmp, patch('vid2player.utils.ball_outgoing_generation.simulate_tables', fake_sim):
            result = generate_tables(gen, tmp, PARAMS)
            self.assertEqual(result['rows'], 18)
            self.assertEqual(len(calls), 20)
            x, y = np.load(Path(tmp)/'ball_traj_out_x.npy'), np.load(Path(tmp)/'ball_traj_out_y.npy')
            expected = decode_rows(np.arange(18), PARAMS)
            np.testing.assert_array_equal(x[:,0], expected[:,0])
            np.testing.assert_array_equal(y[:,0], expected[:,1:])
            self.assertFalse(list(Path(tmp).glob('*.incomplete.npy')))
            with self.assertRaises(FileExistsError):
                generate_tables(gen, tmp, PARAMS)

    def test_failure_keeps_incomplete_files_only(self):
        gen = SimpleNamespace(num_env=5, physics_dt=1/60)
        with tempfile.TemporaryDirectory() as tmp, patch('vid2player.utils.ball_outgoing_generation.simulate_tables', side_effect=ValueError('short flight')):
            with self.assertRaises(ValueError):
                generate_tables(gen, tmp, PARAMS)
            self.assertFalse((Path(tmp)/'ball_traj_out_x.npy').exists())
            self.assertTrue((Path(tmp)/'ball_traj_out_x.incomplete.npy').exists())


if __name__ == '__main__':
    unittest.main()
