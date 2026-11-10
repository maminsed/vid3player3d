"""Live evaluation controls. Importing this module does not load Isaac Gym."""
import time


class EvaluationViewer:
    """Gate entire policy/control steps while keeping the native camera live."""

    def __init__(self, physics, fps, gymapi):
        self.physics = physics
        self.gym = physics.gym
        self.viewer = physics.viewer
        self.gymapi = gymapi
        self.fps = fps
        self.paused = False
        self.closed = False
        self.pending_steps = 0
        self.speed = 1.
        self.next_step_at = 0.
        self.render_at = 0.
        self.camera_keys = set()
        self.preset_keys = set()
        self.camera_tick = time.monotonic()
        self.metrics_line = 'step=0 | t=0.000000s | No control step completed.'
        for key, action in ((gymapi.KEY_SPACE, 'eval_pause'),
                            (gymapi.KEY_RIGHT, 'eval_step'),
                            (gymapi.KEY_MINUS, 'eval_slower'),
                            (gymapi.KEY_EQUAL, 'eval_faster'),
                            (gymapi.KEY_1, 'eval_camera_back'),
                            (gymapi.KEY_2, 'eval_camera_right'),
                            (gymapi.KEY_3, 'eval_camera_left'),
                            (gymapi.KEY_W, 'eval_move_forward'),
                            (gymapi.KEY_S, 'eval_move_back'),
                            (gymapi.KEY_A, 'eval_move_left'),
                            (gymapi.KEY_D, 'eval_move_right')):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, action)
        print('Viewer: Space pause/resume | Right advance one control step (pauses) | '
              '-/= slower/faster | 1/2/3 close back/right/left camera | '
              'WASD move camera, right-drag look | Escape/close finish and save. Starts running.')

    def update_metrics(self, row, events, registered_total, evidence_total, last_registered):
        def number(value, precision):
            return 'n/a' if value is None else f'{value:.{precision}f}'
        registered = sum(event['registered'] for event in events)
        last = 'none' if last_registered is None else (
            f"#{last_registered['id']} @ {last_registered['time_end_s']:.6f}s")
        self.metrics_line = (
            f"step={row['step']} | t={row['time_s']:.6f}s | "
            f"reward={number(row['reward'], 5)} | total_reward={number(row['cumulative_reward'], 5)} | "
            f"distance_now={number(row['ball_racket_distance_m'], 4)}m | "
            f"registered_this_step={registered} | registered_total={registered_total} | "
            f"contact_latched={row['has_contact']} | evidence_intervals_total={evidence_total} | "
            f"last_registered={last}")
        print(self.metrics_line, flush=True)
        for event in events:
            print(f"  Contact evidence #{event['id']}: {event['status']} | "
                  f"t=[{event['time_start_s']:.6f}, {event['time_end_s']:.6f}]s | "
                  f"physics_interval={event['physics_interval']} | "
                  f"sensor={event.get('sensor_evidence')} impulse={event.get('impulse_evidence')}", flush=True)

    def _move_camera_if_native_stationary(self, before, dt):
        pose = self.gym.get_viewer_camera_transform(self.viewer, None)
        # Native right-mouse/WASD gets first chance. Only supply translation
        # when it did not move the camera, avoiding double movement.
        if sum((getattr(pose.p, axis) - getattr(before.p, axis)) ** 2 for axis in 'xyz') > 1e-12:
            return
        forward = pose.r.rotate(self.gymapi.Vec3(0., 0., 1.))
        norm = max((forward.x ** 2 + forward.y ** 2) ** .5, 1e-8)
        right = (forward.y / norm, -forward.x / norm, 0.)
        f = int('eval_move_forward' in self.camera_keys) - int('eval_move_back' in self.camera_keys)
        r = int('eval_move_right' in self.camera_keys) - int('eval_move_left' in self.camera_keys)
        scale = dt / max((f*f + r*r) ** .5, 1.)  # One meter/second, independent of playback speed.
        direction = (forward.x * f + right[0] * r, forward.y * f + right[1] * r, forward.z * f)
        eye = tuple(getattr(pose.p, axis) + scale * direction[i] for i, axis in enumerate('xyz'))
        target = tuple(eye[i] + getattr(forward, axis) for i, axis in enumerate('xyz'))
        self.gym.viewer_camera_look_at(self.viewer, None, self.gymapi.Vec3(*eye), self.gymapi.Vec3(*target))

    def set_camera(self, view):
        # Court-relative directions: +Y toward the opponent, +X to the right.
        # Snap once at the current player position; preserve manual navigation
        # afterward. Pelvis + 25 cm gives a mid-body view without aiming at feet.
        x, y, z = self.physics._root_pos[0].detach().cpu().tolist()
        height = z + .25
        dx, dy = {'back': (0., -2.), 'right': (2., 0.), 'left': (-2., 0.)}[view]
        self.gym.viewer_camera_look_at(
            self.viewer, None, self.gymapi.Vec3(x + dx, y + dy, height),
            self.gymapi.Vec3(x, y, height))

    def render(self):
        if self.closed:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            self.closed = True
            return
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action.startswith('eval_move_'):
                if event.value > 0:
                    self.camera_keys.add(event.action)
                else:
                    self.camera_keys.discard(event.action)
                continue
            if event.action.startswith('eval_camera_'):
                if event.value <= 0:
                    self.preset_keys.discard(event.action)
                elif event.action not in self.preset_keys:
                    self.preset_keys.add(event.action)
                    self.set_camera(event.action[len('eval_camera_'):])
                continue
            if event.value <= 0:
                continue
            if event.action == 'QUIT':
                self.closed = True
                return
            if event.action == 'eval_pause':
                self.paused = not self.paused
                self.pending_steps = 0
                self.next_step_at = 0.
                print('Viewer paused.' if self.paused else 'Viewer running.')
                if self.paused:
                    print('Paused: ' + self.metrics_line, flush=True)
            elif event.action == 'eval_step':
                self.paused = True
                self.pending_steps += 1
                print('Viewer: advancing one control step, then paused.')
            elif event.action in ('eval_slower', 'eval_faster'):
                factor = .5 if event.action == 'eval_slower' else 2.
                self.speed = min(4., max(.125, self.speed * factor))
                self.next_step_at = 0.
                print(f'Viewer target speed: {self.speed:g}x (limited by simulation/rendering).')
        now = time.monotonic()
        dt = min(max(now - self.camera_tick, 0.), .1)
        self.camera_tick = now
        before = self.gym.get_viewer_camera_transform(self.viewer, None) if self.camera_keys else None
        self.gym.step_graphics(self.physics.sim)
        self.gym.draw_viewer(self.viewer, self.physics.sim, False)
        if before is not None:
            self._move_camera_if_native_stationary(before, dt)

    def wait_for_step(self):
        # Draw once at each control boundary, plus at most 30 Hz when paused
        # or pacing. No policy inference, resets, or physics run in this loop.
        self.render_at = 0.
        while not self.closed:
            now = time.monotonic()
            if now >= self.render_at:
                self.render()
                self.render_at = time.monotonic() + 1 / 30
            if self.closed:
                return False
            if self.paused:
                if self.pending_steps:
                    self.pending_steps -= 1
                    return True
            elif now >= self.next_step_at:
                self.next_step_at = now + 1 / (self.fps * self.speed)
                return True
            time.sleep(.005)
        return False
