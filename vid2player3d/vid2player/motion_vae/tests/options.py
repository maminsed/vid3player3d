"""Self-contained test options, independent of named experiment configs."""
from vid2player.motion_vae.config import MotionVAEOption


def motion_options(predict_phase=False):
    opt = MotionVAEOption()
    opt.update(model_ver='test_motion', frame_size=288, predict_phase=predict_phase,
               mixed_phase_schedule=None, softmax_future=True,
               weights={'recon': 1, 'kl': .5, 'recon_phase': 10, 'trajectory': 0})
    return opt
