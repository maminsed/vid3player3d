"""CPU coverage for case selection, recording and the evaluation loop."""
import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import imageio.v2 as imageio

from vid2player.utils.controller_evaluation import (
    configure_eval_args, prepare_evaluation, select_ball_case, CourtRenderer, evaluate_controller)


def offline_generator_class():
    # Avoid importing IsaacGym/GUI dependencies for CPU sampler checks.
    path = Path(__file__).resolve().parents[1]/'utils/tennis_ball.py'
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node,ast.ClassDef) and node.name=='TennisBallGeneratorOffline']
    namespace = {'torch':torch,'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['TennisBallGeneratorOffline']


class ControllerEvaluationTests(unittest.TestCase):
    def test_offline_sampler_device_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            data=np.arange(6*307,dtype=np.float32).reshape(6,307)
            filename=Path(directory)/'balls.npy';np.save(filename,data)
            generator=offline_generator_class()(filename,sample_random=False,num_envs=6)
            generator.sample_idx[:]=torch.arange(6)
            conversions=[]
            # A device-boundary sentinel verifies explicit conversion even on a
            # CPU host: direct indexing with this object fails like CUDA IDs.
            class DeviceIds:
                def detach(self):
                    return self
                def to(self,device,dtype):
                    conversions.append((device,dtype))
                    return torch.tensor([4,1],device=device,dtype=dtype)
            result=generator.generate(2,True,env_ids=DeviceIds())
            self.assertEqual(conversions,[(torch.device('cpu'),torch.long)])
            np.testing.assert_array_equal(result[0].numpy(),data[[4,1],7:].reshape(2,100,3))
            np.testing.assert_array_equal(result[1].numpy(),data[[4,1],:3])
            self.assertEqual(generator.sample_idx.tolist(),[0,2,2,3,5,5])
            generator.generate(1,env_ids=torch.tensor([5]))
            self.assertEqual(generator.sample_idx[5].item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_offline_sampler_cuda_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data=np.arange(6*307,dtype=np.float32).reshape(6,307)
            filename=Path(directory)/'balls.npy';np.save(filename,data)
            cls=offline_generator_class()
            generator=cls(filename,sample_random=False,num_envs=6)
            generator.sample_idx[:]=torch.arange(6)
            result=generator.generate(2,True,env_ids=torch.tensor([4,1],device='cuda'))
            np.testing.assert_array_equal(result[0].numpy(),data[[4,1],7:].reshape(2,100,3))
            self.assertTrue(all(value.device.type=='cpu' for value in result))
            generator=cls(filename,sample_random=True,num_envs=6)
            generator.generate(2,True,start_pos=torch.tensor([[0.,-13.,1.],[0.,13.,1.]],device='cuda'))

    def test_case_selection_and_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.zeros((6,307), dtype=np.float32)
            for i in range(6):
                traj = np.zeros((100,3), dtype=np.float32)
                traj[:,0],traj[:,1],traj[:,2] = [-2,0,2][i//2],np.linspace(13,-15,100),[.9,1.4][i%2]
                data[i,:7] = [0,13,1,0,-15,1,5]
                data[i,7:] = traj.ravel()
            path = root/'balls.npy';np.save(path,data)
            for i,name in enumerate(['left_low','left_high','center_low','center_high','right_low','right_high']):
                rows,case = select_ball_case(path,name)
                self.assertEqual(case['source_row'],i)
                np.testing.assert_array_equal(rows,data[i:i+1])
            _,case = select_ball_case(path,'left_low',x=2.,height=1.4)
            self.assertEqual(case['source_row'],5)
            args = SimpleNamespace(eval_checkpoint=str(root/'policy.pth'),eval_steps=4,
                seed=None,eval_output=str(root/'eval'),cfg='test',eval_ball='center_low',
                eval_ball_x=None,eval_ball_height=None,eval_target='center')
            (root/'policy.pth').write_bytes(b'placeholder')
            configure_eval_args(args)
            self.assertTrue(args.headless and args.test and not args.train)
            self.assertEqual(args.num_envs,1)
            cfg = {'name':'MVAEControllerVis','env':{'vid2player':{'ball_traj_file':str(path)}}}
            training = {'params':{'config':{}}}
            output,case,temporary = prepare_evaluation(args,cfg,training)
            self.assertEqual(cfg['name'],'PhysicsMVAEController')
            self.assertEqual([x.name for x in output.iterdir()],['evaluation_config.json'])
            self.assertTrue(Path(temporary.name,'launch.npy').is_file())
            temporary.cleanup()
            with self.assertRaises(FileExistsError):
                prepare_evaluation(args,cfg,training)
            args.eval_ball='invalid'
            with self.assertRaises(ValueError):
                configure_eval_args(args)

    def test_mvae_camera_equivalence(self):
        path=Path(__file__).resolve().parents[1]/'motion_vae/reconstruction.py'
        nodes=[node for node in ast.parse(path.read_text()).body
               if isinstance(node,ast.ClassDef) and node.name=='CourtCamera']
        ns={'np':np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        mvae=ns['CourtCamera']()
        camera=CourtRenderer(['Pelvis'],[-5,-16],[5,-10])
        points=np.array([[0,-13,1],[-5,-11.89,0],[5,11.89,0],[0,0,1]])
        np.testing.assert_allclose(camera.project(points),mvae.project(points)[0])
        self.assertEqual(camera.background.size,(768,720))

    def test_recording_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'eval';output.mkdir()
            model=torch.nn.Linear(1,1)
            checkpoint=root/'policy.pth';torch.save({'model':model.state_dict()},checkpoint)
            source=root/'source.npy';np.save(source,np.zeros((1,307),dtype=np.float32))
            (output/'evaluation_config.json').write_text('{}')
            args=SimpleNamespace(eval_checkpoint=str(checkpoint),eval_steps=5,seed=123,eval_target='right')
            physics=SimpleNamespace(dt=1/30,humanoid_asset=object(),
                gym=SimpleNamespace(get_asset_rigid_body_names=lambda _:['Pelvis','Torso','Spine','Chest','Neck','Head']),
                _rigid_body_pos=torch.zeros(1,6,3),_has_racket_ball_contact=torch.zeros(1,dtype=torch.bool),
                _has_bounce_now=torch.zeros(1,dtype=torch.bool),_bounce_pos=torch.zeros(1,3),
                _ball_generator=SimpleNamespace(sample_idx=torch.zeros(1,dtype=torch.long)))
            task=SimpleNamespace(device='cpu',_physics_player=SimpleNamespace(task=physics),
                cfg_v2p={'court_min':[-5,-16],'court_max':[5,-10],
                         'reward_weights':{'pos':1.},'aux_loss_specs':{'dof_res':.01}},
                _num_mvae_action=32,_num_res_dof_action=3,
                _reset_reaction_buf=torch.ones(1,dtype=torch.bool),_reset_recovery_buf=torch.zeros(1,dtype=torch.bool),
                _tar_time_total=torch.zeros(1,dtype=torch.long),_target_bounce_pos=torch.zeros(1,3),
                _ball_pos=torch.zeros(1,3),_racket_pos=torch.zeros(1,3),_racket_normal=torch.zeros(1,3),
                _est_bounce_in=torch.zeros(1,dtype=torch.bool),_root_pos=torch.tensor([[0.,-13.,1.]]),
                obs_buf=torch.zeros(1,1),_compute_observations=lambda:None)
            task._racket_normal[:,1]=1
            physics._rigid_body_pos[:,:,1]=-13
            physics._rigid_body_pos[:,:,2]=torch.linspace(.8,1.8,6)
            task._racket_pos[:,1:]=torch.tensor([-13.,1.])
            steps=[];resets=[];cleanups=[]
            def reset(ids):
                resets.append(ids.tolist())
                return {'obs':task.obs_buf}
            def step(env,action):
                n=len(steps)+1;steps.append(n)
                task._ball_pos[:,1:]=torch.tensor([-13.,1.])
                task._ball_pos[:,0]=.4/n
                physics._has_racket_ball_contact[:]=n>=2
                task._est_bounce_in[:]=n>=2
                done=torch.tensor([n>=4])
                if n>=4:
                    task._root_pos[0,0]=6
                task._reset_reaction_buf[:]=done
                return {'obs':task.obs_buf},torch.tensor([[.1]]),done,{
                    'sub_rewards':torch.tensor([[.1]]),'sub_rewards_names':'pos_reward'}
            means=torch.zeros(1,35);means[0,32]=.2;means[0,0]=1.1
            player=SimpleNamespace(device='cpu',model=model,normalize_input=False,task=task,is_rnn=False,
                config={'bounds_loss_coef':10.},env=object(),env_reset=reset,get_batch_size=lambda obs,n:n,
                obs_to_torch=lambda obs:{'obs':obs},get_action=lambda *a,**k:means,env_step=step)
            case={'preset':'right_high','source_row':0,'source_file':str(source)}
            temporary=SimpleNamespace(cleanup=lambda:cleanups.append(True))
            config_stub=SimpleNamespace(set_seed=lambda *args:torch.manual_seed(args[0]))
            with patch.dict(sys.modules,{'vid2player.utils.config':config_stub}):
                evaluate_controller(player,args,(output,case,temporary))
            report=json.loads((output/'evaluation_config.json').read_text())['evaluation_results']
            data=np.load(output/'rollout.npz')
            first=dict(zip(data['metric_names'].tolist(),data['metrics'][0]))
            self.assertEqual(report['steps'],4)
            self.assertEqual(report['contact_step'],2)
            self.assertEqual(report['end_reason'],'movement_boundary')
            self.assertAlmostEqual(report['total_reward'],.4)
            self.assertAlmostEqual(first['aux_dof_res_loss'],.04)
            self.assertAlmostEqual(first['weighted_aux_dof_res_loss'],.0004)
            self.assertAlmostEqual(first['bounds_loss'],.01,places=6)
            self.assertAlmostEqual(first['weighted_bounds_loss'],.1,places=6)
            self.assertAlmostEqual(first['total_regularization_loss'],.1004,places=6)
            self.assertAlmostEqual(first['weighted/pos_reward'],.1)
            self.assertEqual(resets[0],[0])
            self.assertTrue(all(not ids for ids in resets[1:]))
            self.assertEqual(cleanups,[True])
            self.assertEqual({p.name for p in output.iterdir()},
                {'video.mp4','evaluation_config.json','rollout.npz','metrics.png'})
            np.testing.assert_array_equal(task._target_bounce_pos.numpy(),[[3.,10.,0.]])
            reader=imageio.get_reader(output/'video.mp4')
            self.assertEqual(reader.get_data(0).shape,(720,768,3))
            reader.close()
            from PIL import Image
            with Image.open(output/'metrics.png') as plot:
                self.assertEqual(plot.size,(1800,1200))


if __name__=='__main__':
    unittest.main()
