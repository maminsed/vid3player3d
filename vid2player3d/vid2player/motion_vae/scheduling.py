"""Epoch-based curriculum for the number of frames in a training window."""
import math
from numbers import Integral, Real


def sequence_length(opt, epoch):
    """Interpolate (epoch fraction, frames) knots; round down and hold endpoints."""
    minimum = opt.num_condition_frames + opt.num_future_predictions
    schedule = opt.nframes_seq_schedule
    if schedule is None:
        return opt.nframes_seq
    error = ('nframes_seq_schedule requires at least two (epoch_fraction, frames) knots, '
             'strictly increasing fractions in [0,1], and nondecreasing integer lengths '
             '>= num_condition_frames + num_future_predictions.')
    if not isinstance(schedule, (list, tuple)) or len(schedule) < 2:
        raise ValueError(error)
    previous_fraction, previous_frames = -1, minimum
    for knot in schedule:
        if not isinstance(knot, (list, tuple)) or len(knot) != 2:
            raise ValueError(error)
        fraction, frames = knot
        if (isinstance(fraction, bool) or not isinstance(fraction, Real)
                or not math.isfinite(fraction) or not 0 <= fraction <= 1
                or fraction <= previous_fraction
                or isinstance(frames, bool) or not isinstance(frames, Integral)
                or frames < previous_frames):
            raise ValueError(error)
        previous_fraction, previous_frames = fraction, frames
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    if total_epochs <= 0:
        raise ValueError('nframes_seq_schedule requires a positive total epoch count.')
    progress = epoch / total_epochs
    if progress <= schedule[0][0]:
        return schedule[0][1]
    for (lo, start), (hi, end) in zip(schedule, schedule[1:]):
        if progress < hi:
            return math.floor(start + (end - start) * (progress - lo) / (hi - lo) + 1e-9)
    return schedule[-1][1]
