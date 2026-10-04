"""The dashboard is an explicit small allowlist, never a dump of metric fields."""
import unittest

from vid2player.motion_vae.monitoring import train_tag, validation_scalars


class MonitoringTests(unittest.TestCase):
    def test_primary_cards_are_autoregressive_means_and_average_both_wrists(self):
        auto = dict(recon_weighted_mean=2., kl_weighted_mean=.1, root_error_m_mean=.3,
                    local_mpjpe_m_mean=.04, root_endpoint_error_m_mean=.5,
                    left_wrist_error_m_mean=.08, right_wrist_error_m_mean=.12, failed_rollouts=0)
        teacher = dict(auto, recon_weighted_mean=1., recon_mean=1., recon_zero_latent_mean=1.25,
                       latent_joint_effect_m_mean=.01, root_error_m_mean=.02)
        result = validation_scalars({'window_100/autoregressive': auto,
                                     'window_100/teacher_forced': teacher,
                                     'whole_sequence/autoregressive': auto,
                                     'whole_sequence/teacher_forced': teacher})
        self.assertEqual(result['val_100/recon_loss'], 2.)
        self.assertEqual(result['val_full/root_error_m'], .3)
        self.assertAlmostEqual(result['val_100/wrist_error_m'], .1)
        self.assertEqual(result['diagnostics/val_100/one_step_root_error_m'], .02)
        self.assertEqual(result['diagnostics/val_full/latent_recon_benefit'], .25)
        self.assertEqual(len([k for k in result if not k.startswith('diagnostics/')]), 12)
        self.assertEqual(train_tag('recon'), 'train/recon_loss')
        self.assertEqual(train_tag('kl'), 'train/kl_loss')

    def test_flat_model_results_remain_but_static_facts_and_unknown_metrics_do_not(self):
        values = dict(kl_weighted_mean=0., failed_rollouts=0,
                      predicted_frames=100, rollouts=3, stationary_frames=10,
                      gt_root_speed_mps_mean=.5, gt_wrist_speed_mps_mean=2.,
                      fast_motion_frames=20, arbitrary_new_metric=42.)
        result = validation_scalars({'whole_sequence/autoregressive': values})
        self.assertEqual(result, {'val_full/kl_loss': 0.,
                                  'diagnostics/val_full/autoregressive_failed_rollouts': 0})
        values['failed_rollouts'] = 2
        self.assertEqual(validation_scalars({'whole_sequence/autoregressive': values})[
            'diagnostics/val_full/autoregressive_failed_rollouts'], 2)


if __name__ == '__main__':
    unittest.main()
