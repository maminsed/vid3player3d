"""Reward curricula measured in collected environment transitions."""
import math


class LinearRewardSchedule:
    """Hold the start value, interpolate, then hold the end value."""

    def __init__(self, config):
        if config.get('type', 'linear') != 'linear':
            raise ValueError('Reward schedule type must be linear.')
        self.start = float(config['start'])
        self.end = float(config['end'])
        self.start_frame = int(config['start_frame'])
        self.end_frame = int(config['end_frame'])
        if not all(math.isfinite(x) and x > 0 for x in (self.start, self.end)):
            raise ValueError('Reward schedule scales must be finite and positive.')
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError('Reward schedule requires 0 <= start_frame < end_frame.')

    def value(self, frame):
        fraction = min(1., max(0., (frame - self.start_frame) /
                              (self.end_frame - self.start_frame)))
        return self.start + fraction * (self.end - self.start)
