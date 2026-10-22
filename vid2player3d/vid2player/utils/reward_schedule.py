"""Reward curricula measured in collected environment transitions."""
import math


class LinearRewardSchedule:
    """Hold the start value, interpolate, then hold the end value."""

    def __init__(self, config, allow_zero=False):
        if config.get('type', 'linear') != 'linear':
            raise ValueError('Reward schedule type must be linear.')
        self.start = float(config['start'])
        self.end = float(config['end'])
        self.start_frame = int(config['start_frame'])
        self.end_frame = int(config['end_frame'])
        if not all(math.isfinite(x) and (x >= 0 if allow_zero else x > 0)
                   for x in (self.start, self.end)):
            raise ValueError('Reward schedule endpoints must be finite and ' +
                             ('nonnegative.' if allow_zero else 'positive.'))
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError('Reward schedule requires 0 <= start_frame < end_frame.')

    def value(self, frame):
        fraction = min(1., max(0., (frame - self.start_frame) /
                              (self.end_frame - self.start_frame)))
        return self.start + fraction * (self.end - self.start)


class PiecewiseRewardSchedule:
    """Ordered linear ramps with holds between them and continuous endpoints."""

    def __init__(self, config, allow_zero=False):
        specs = config.get('segments')
        if not isinstance(specs, list) or not specs:
            raise ValueError('Piecewise reward schedule requires a nonempty segments list.')
        if any(key in config for key in ('start', 'end', 'start_frame', 'end_frame', 'type')):
            raise ValueError('Put linear ramp settings inside segments, not beside them.')
        self.segments = []
        for spec in specs:
            if not isinstance(spec, dict) or 'enabled' in spec:
                raise ValueError('Each segment must be a linear mapping; set enabled on the whole schedule.')
            ramp = LinearRewardSchedule(spec, allow_zero=allow_zero)
            if self.segments:
                previous = self.segments[-1]
                if ramp.start_frame < previous.end_frame:
                    raise ValueError('Reward schedule segments must be ordered and nonoverlapping.')
                if not math.isclose(ramp.start, previous.end, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError('Consecutive reward schedule segments must have matching end/start values.')
            self.segments.append(ramp)

    def value(self, frame):
        value = self.segments[0].start
        for ramp in self.segments:
            if frame < ramp.start_frame:
                break
            value = ramp.value(frame)
        return value
