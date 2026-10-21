"""Config-folder contracts without importing IsaacGym or running training."""
import ast
from functools import partial
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from vid2player.tools.regenerate_ball_previews import load_config
from vid2player.utils.config_paths import resolve_config, config_output_name
from vid2player.utils.training_io import check_training_output
from vid2player.utils.controller_evaluation import prepare_evaluation
from vid2player.utils.controller_validation import prepare_validation


SOURCE = Path(__file__).resolve().parents[1]


class ConfigPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'cfg'
        self.path = self.root / 'controller/strategy/deeper/stage.yaml'
        self.path.parent.mkdir(parents=True)
        self.path.write_text((SOURCE / 'cfg/controller/usopen_train_stage_1.yaml').read_text())

    def test_qualified_names_extensions_and_paths_agree(self):
        for value in ('stage', 'stage.yaml', 'strategy/deeper/stage',
                      'strategy/deeper/stage.yaml', 'controller/strategy/deeper/stage',
                      str(self.path)):
            with self.subTest(value=value):
                resolved = resolve_config(value, self.root)
                self.assertEqual(resolved, self.path)
                self.assertEqual(config_output_name(resolved, self.root), 'strategy__deeper__stage')

    def test_duplicate_basenames_require_qualification(self):
        other = self.root / 'controller/stage.yaml'
        other.write_text(self.path.read_text())
        with self.assertRaisesRegex(ValueError, 'Ambiguous.*controller/stage.yaml'):
            resolve_config('stage', self.root)
        self.assertEqual(resolve_config('strategy/deeper/stage', self.root), self.path)
        self.assertEqual(resolve_config('controller/stage', self.root), other)
        with self.assertRaises(FileNotFoundError):
            resolve_config('wrong_folder/stage', self.root)
        with self.assertRaises(FileNotFoundError):
            resolve_config(self.root / 'missing.yaml', self.root)

    def test_legacy_and_external_names(self):
        for category in ('controller', 'im'):
            path = self.root / category / 'legacy.yaml'
            path.parent.mkdir(exist_ok=True)
            path.touch()
            self.assertEqual(config_output_name(path, self.root), 'legacy')
        external = Path(self.temporary.name) / 'external.yaml'
        external.write_text(self.path.read_text())
        self.assertEqual(resolve_config(external, self.root), external)
        self.assertEqual(config_output_name(external, self.root), 'external')

    def test_preview_uses_same_lookup_and_flat_name(self):
        with patch('vid2player.tools.regenerate_ball_previews.resolve_config',
                   partial(resolve_config, root=self.root)):
            path, cfg = load_config('strategy/deeper/stage')
            self.assertEqual(load_config(str(path)), (path, cfg))
            self.assertEqual(config_output_name(path, self.root), 'strategy__deeper__stage')

    def test_training_load_preserves_selector_and_flattens_output(self):
        # Run the actual loader with CPU helpers, without importing IsaacGym.
        tree = ast.parse((SOURCE / 'utils/config.py').read_text())
        get_args = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_args')
        parameters = eval(compile(ast.Expression(get_args.body[0].value), '<parameters>', 'eval'))
        defaults = {p['name'][2:].replace('-', '_'): p.get('default', False if p.get('action') == 'store_true' else None)
                    for p in parameters}
        defaults.update(cfg='strategy/deeper/stage', train=True, torch_deterministic=False,
                        results_dir=str(Path(self.temporary.name) / 'results'))
        args = SimpleNamespace(**defaults)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_cfg')
        namespace = dict(os=os, yaml=yaml, resolve_config=partial(resolve_config, root=self.root),
                         config_output_name=partial(config_output_name, root=self.root),
                         check_training_output=check_training_output, set_seed=lambda seed, deterministic: seed)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<load_cfg>', 'exec'), namespace)
        for selector in ('strategy/deeper/stage', 'stage', str(self.path)):
            with self.subTest(selector=selector):
                args.cfg = selector
                cfg, training = namespace['load_cfg'](args)
                self.assertEqual(args.cfg, selector)
                self.assertEqual(args.cfg_path, str(self.path))
                self.assertEqual(args.cfg_name, 'strategy__deeper__stage')
                output = Path(args.results_dir) / args.cfg_name
                self.assertEqual(cfg['cfg_dir'], str(output))
                self.assertEqual(cfg['env']['controller_cfg_name'], args.cfg_name)
                for key, folder in (('network_path', 'models'), ('log_path', 'logs'), ('train_dir', 'train')):
                    self.assertEqual(training['params']['config'][key], str(output / folder))
                self.assertTrue((output / 'models').is_dir())
                self.assertTrue((output / 'logs').is_dir())
                self.assertFalse(cfg['env']['vid2player']['legacy_eval_residual_gating'])
        # Exercise the CLI option's actual argparse definition and config propagation.
        import argparse
        parser = argparse.ArgumentParser()
        flag = next(p.copy() for p in parameters if p['name'] == '--env-flag')
        parser.add_argument(flag.pop('name'), **flag)
        args.env_flag = parser.parse_args(['--env-flag', 'legacy_eval_residual_gating']).env_flag
        cfg, _ = namespace['load_cfg'](args)
        self.assertTrue(cfg['env']['vid2player']['legacy_eval_residual_gating'])
        args.env_flag = ['unknown_option']
        with self.assertRaisesRegex(ValueError, 'Unknown --env-flag'):
            namespace['load_cfg'](args)

    def test_default_evaluation_and_validation_directories_are_flat(self):
        args = SimpleNamespace(cfg='strategy/deeper/stage', cfg_name='strategy__deeper__stage',
                               eval_output=None, eval_checkpoint='policy.pth', eval_steps=2,
                               eval_ball='center_low', eval_ball_x=None, eval_ball_height=None,
                               eval_target='center', seed=123, num_envs=1, rl_device='cpu',
                               validation_shots_per_preset=1)
        rows = np.zeros((1, 307), dtype=np.float32)
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.temporary.name)
        for prepare, selection, result, parent in (
                (prepare_evaluation, 'controller_evaluation.select_ball_case', (rows, {}), 'controller_eval'),
                (prepare_validation, 'controller_validation.select_validation_balls', (rows, [{}]), 'controller_validation')):
            cfg = {'name': 'PhysicsMVAEController', 'env': {
                'episodeLength': 10, 'vid2player': {'ball_traj_file': 'balls.npy'}}}
            with patch('vid2player.utils.' + selection, return_value=result):
                prepared = prepare(args, cfg, {'params': {'config': {}}})
            output, temporary = prepared[0], prepared[-1]
            self.addCleanup(temporary.cleanup)
            self.assertEqual(output.parent, Path('out') / parent)
            self.assertTrue(output.name.startswith('strategy__deeper__stage_policy_'))
            self.assertTrue(output.is_dir())


if __name__ == '__main__':
    unittest.main()
