"""Exercise playback gating without a display, GPU, or Isaac Gym import."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from vid2player.utils.controller_viewer import EvaluationViewer


class ViewerTests(unittest.TestCase):
    def make_viewer(self, batches=()):
        self.now = 1.
        self.frames = 0
        self.batches = list(batches)
        api = NS(KEY_SPACE=1, KEY_RIGHT=2, KEY_MINUS=3, KEY_EQUAL=4,
                 KEY_1=5, KEY_2=6, KEY_3=7, KEY_W=8, KEY_S=9, KEY_A=10, KEY_D=11,
                 Vec3=lambda x,y,z: (x,y,z))
        self.window_closed = False
        gym = NS(subscribe_viewer_keyboard_event=lambda *a: None,
                 query_viewer_has_closed=lambda _: self.window_closed,
                 query_viewer_action_events=lambda _: self.batches.pop(0) if self.batches else [],
                 step_graphics=lambda _: None, draw_viewer=self.draw)
        return EvaluationViewer(NS(gym=gym, viewer=object(), sim=object()), 30, api)

    def draw(self, *args):
        self.frames += 1

    def sleep(self, seconds):
        self.now += seconds
        self.assertLess(self.now, 10., 'Playback gating did not return')

    def wait(self, viewer):
        with patch('vid2player.utils.controller_viewer.time.monotonic', lambda: self.now), \
             patch('vid2player.utils.controller_viewer.time.sleep', self.sleep):
            return viewer.wait_for_step()

    @staticmethod
    def event(action, value=1):
        return NS(action=action, value=value)

    def test_starts_running_and_paces_without_catchup(self):
        viewer = self.make_viewer()
        self.assertTrue(self.wait(viewer))
        self.assertFalse(viewer.paused)
        self.assertEqual(self.frames, 1)
        start = self.now
        self.assertTrue(self.wait(viewer))
        self.assertGreaterEqual(self.now - start, 1/30)
        # Expensive inference/drawing must not accumulate a burst of catch-up steps.
        self.now += 2
        self.assertTrue(self.wait(viewer))
        self.assertAlmostEqual(viewer.next_step_at, self.now + 1/30)

    def test_pause_keeps_camera_rendering_and_single_step_remains_paused(self):
        viewer = self.make_viewer([[self.event('eval_pause')], [], [self.event('eval_step')]])
        # No control step is released until the single-step event arrives.
        self.assertTrue(self.wait(viewer))
        self.assertGreaterEqual(self.frames, 3)
        self.assertTrue(viewer.paused)
        self.assertEqual(viewer.pending_steps, 0)
        viewer.gym.query_viewer_action_events = lambda _: [self.event('QUIT')]
        self.assertFalse(self.wait(viewer))

    def test_resume_clears_steps_and_release_events_are_ignored(self):
        viewer = self.make_viewer([[self.event('eval_step'), self.event('eval_pause'),
                                    self.event('eval_pause', 0)]])
        self.assertTrue(self.wait(viewer))
        self.assertFalse(viewer.paused)
        self.assertEqual(viewer.pending_steps, 0)

    def test_speed_changes_and_close(self):
        viewer = self.make_viewer([[self.event('eval_slower')]])
        self.assertTrue(self.wait(viewer))
        self.assertEqual(viewer.speed, .5)
        self.assertAlmostEqual(viewer.next_step_at - self.now, 1/15)
        self.batches = [[self.event('eval_faster') for _ in range(10)]]
        self.assertTrue(self.wait(viewer))
        self.assertEqual(viewer.speed, 4.)
        self.window_closed = True
        self.assertFalse(self.wait(viewer))

    def test_close_camera_presets_use_current_player_and_work_while_paused(self):
        viewer = self.make_viewer()
        viewer.paused = True
        root = [4., -12., .9]
        tensor = NS(tolist=lambda: list(root))
        tensor.detach = tensor.cpu = lambda: tensor
        viewer.physics._root_pos = [tensor]
        camera_calls = []
        viewer.gym.viewer_camera_look_at = lambda *args: camera_calls.append(args[2:])
        expected = {'back': (4., -14., 1.15), 'right': (6., -12., 1.15), 'left': (2., -12., 1.15)}
        for name, eye in expected.items():
            self.batches = [[self.event('eval_camera_' + name), self.event('eval_camera_' + name, 0)]]
            viewer.render()
            self.assertEqual(camera_calls[-1], (eye, (4., -12., 1.15)))
            self.assertTrue(viewer.paused)
            self.assertEqual(viewer.pending_steps, 0)
        root[:] = [0., -10., 1.]
        self.batches = [[self.event('eval_camera_back')]]
        viewer.render()
        self.assertEqual(camera_calls[-1], ((0., -12., 1.25), (0., -10., 1.25)))

    def test_preset_does_not_repeat_and_keyboard_fallback_preserves_look_direction(self):
        viewer = self.make_viewer()
        presets = []
        viewer.set_camera = lambda name: presets.append(name)
        forward = NS(x=0., y=1., z=0.)
        pose = NS(p=NS(x=0., y=-2., z=1.), r=NS(rotate=lambda _: forward))
        viewer.gym.get_viewer_camera_transform = lambda *a: pose
        moves = []
        viewer.gym.viewer_camera_look_at = lambda *a: moves.append(a[2:])
        viewer.camera_tick = 1.
        self.batches = [[self.event('eval_camera_back'), self.event('eval_move_forward')],
                        [self.event('eval_camera_back')],
                        [self.event('eval_camera_back', 0), self.event('eval_move_forward', 0)]]
        with patch('vid2player.utils.controller_viewer.time.monotonic', side_effect=[1.05,1.1,1.15]):
            viewer.render()
            viewer.render()
            viewer.render()
        self.assertEqual(presets, ['back'])
        self.assertEqual(len(moves), 2)
        self.assertAlmostEqual(moves[0][0][1], -1.95)
        self.assertAlmostEqual(moves[0][1][1] - moves[0][0][1], 1.)
        self.assertFalse(viewer.camera_keys)
        # If native controls already translated the camera, do not double it.
        viewer.camera_keys.add('eval_move_forward')
        before = NS(p=NS(x=0.,y=-3.,z=1.))
        viewer._move_camera_if_native_stationary(before,.05)
        self.assertEqual(len(moves), 2)

    def test_metrics_print_per_step_and_on_pause_not_on_camera_redraw(self):
        viewer = self.make_viewer()
        row = dict(step=12, time_s=.4, reward=.7, cumulative_reward=1.2,
                   ball_racket_distance_m=.08, has_contact=True)
        events = [dict(id=4, time_start_s=.38, time_end_s=.4, registered=True,
                       physics_interval=1, status='registered')]
        with patch('builtins.print') as output:
            viewer.update_metrics(row, events, 1, 4, events[0])
            self.assertEqual(output.call_count, 2)
            self.batches = [[], [self.event('eval_pause')], []]
            viewer.render()
            self.assertEqual(output.call_count, 2)
            viewer.render()
            self.assertEqual(output.call_count, 4)
            self.assertIn('Paused: step=12 | t=0.400000s', output.call_args.args[0])
            self.assertIn('last_registered=#4 @ 0.400000s', output.call_args.args[0])
            viewer.render()
            self.assertEqual(output.call_count, 4)


if __name__ == '__main__':
    unittest.main()
