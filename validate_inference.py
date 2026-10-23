"""Compare RGMT to pinned author observation, history, network and action code.

uv run validate_inference.py --source PMT --isaac-math math.py
  --isaac-buffer circular_buffer.py --weights rgmt_flat.pt --output parity.json
"""
import argparse
import ast
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from scipy.spatial.transform import Rotation as R
from adapter import PMTRGMT
from motion_challenge.sdk import BodyState, StatePacket, ReferenceFrame
from validation.native_config import source_definitions, validate_control_config


def methods(path, class_name, names, scope):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node.bases=[];node.decorator_list=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[node],type_ignores=[]),str(path),'exec'),scope)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','isaac-math','isaac-buffer','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    pin=json.loads((root/'native-sources.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,expected in pin['files'].items():assert sha(args.source/name)==expected,name
    assert sha(args.isaac_math)==pin['isaaclab_math_sha256']
    assert sha(args.isaac_buffer)==pin['isaaclab_circular_buffer_sha256']
    runtime=json.loads((root/'runtime.json').read_text());runtime['weights']=str(args.weights.resolve());adapter=PMTRGMT(root,runtime)
    # Validate native default/PD settings; RGMT explicitly overrides the base action scale below.
    base_cfg=json.loads((root/'policy.json').read_text());base_cfg['action_scale']=1.
    scope={'torch':torch,'Tensor':torch.Tensor,'nn':nn,'F':F,'np':np,'math':math,'reduce':reduce,'Normal':Normal,
           'json':json,'hashlib':hashlib,'resolve_nn_activation':lambda name:{'elu':nn.ELU}[name]()}
    checks=validate_control_config(args.source,root,adapter,pin)
    cfg_text=(args.source/'pmt_tasks/env_cfgs/rgmt/rgmt.py').read_text()
    tree=ast.parse(cfg_text)
    assignments={ast.unparse(n.targets[0]):ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) in ('self.actions.joint_pos.scale','self.commands.motion.update_action_offset_with_ref','self.commands.motion.anchor_body')}
    assert assignments=={'self.actions.joint_pos.scale':1.0,'self.commands.motion.update_action_offset_with_ref':True,'self.commands.motion.anchor_body':'torso_link'}
    assert base_cfg['action_scale']==1.0 and adapter.cfg['anchor_body']=='torso_link'
    checks['native_rgmt_action_offset_and_anchor']=True
    base=args.source/'motion_tracking_rl'
    for path,names in [(base/'networks/layers/normalization.py',['EmpiricalNormalization']),
                       (base/'networks/layers/mlp.py',['MLP']), (base/'utils/utils.py',['build_obs_schema']),
                       (base/'networks/transformer_actor_critic.py',['TwoLayerMLP','SinusoidalPositionalEncoding','SelfAttentionBlock','CrossAttentionBlock','TransformerActorCritic'])]:
        source_definitions(path,names,scope)
    checkpoint=torch.load(args.weights,map_location='cpu',weights_only=True)
    metadata=checkpoint['policy_metadata'];dummy={k:torch.zeros(1,*v['shape']) for k,v in metadata['obs_schema']['available_obs'].items()}
    native=scope['TransformerActorCritic'](dummy,metadata['obs_schema']['obs_groups'],29,
        actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],actor_obs_normalization=True,
        history_obs_normalization=True,command_obs_normalization=True,critic_obs_normalization=True,
        n_embd=128,n_heads=4,history_len=10,cmd_len=21,mlp_ratio=4).eval()
    native.load_state_dict(checkpoint['model_state_dict'],strict=True)
    assert native._obs_schema==metadata['obs_schema']
    source_definitions(args.isaac_buffer,['CircularBuffer'],scope)
    source_definitions(args.isaac_math,['normalize','quat_conjugate','quat_inv','quat_apply'],scope)
    scope['SceneEntityCfg']=lambda name:NS(name=name,joint_ids=slice(None))
    source_definitions(args.source/'pmt_tasks/mdp/observations.py',['proprio'],scope)
    methods(args.source/'pmt_tasks/mdp/commands/multi_motion_command.py','MultiMotionCommandV2',{'get_command_window','_maybe_update_action_offset'},scope)
    rng=np.random.default_rng(281023);names=tuple(np.asarray(adapter.names)[rng.permutation(29)])
    errors=dict(proprio=0.,history=0.,command=0.,network=0.,action=0.,motor_target=0.,reset_history=0.)
    def err(key,a,b):errors[key]=max(errors[key],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    def tensor(x):return torch.tensor(np.asarray(x),dtype=torch.float32).unsqueeze(0)
    def body():return BodyState('torso_link',tuple(rng.normal(size=3)),tuple(R.random(random_state=rng).as_quat()),tuple(rng.normal(size=3)),tuple(rng.normal(size=3)))
    with torch.inference_mode():
        for reset in range(4):
            history=scope['CircularBuffer'](10,1,'cpu');last=torch.zeros(1,29)
            for step in range(40):
                rotation=R.random(random_state=rng)
                state=StatePacket(step*.02,names,tuple(rng.normal(0,.3,29)),tuple(rng.normal(0,.5,29)),(0,0,.8),tuple(rotation.as_quat()),(0,0,0),tuple(rng.normal(0,.3,3)))
                refs=tuple(ReferenceFrame((step+i)*.02,max(0,step+i)*.02,tuple(rng.normal(0,.3,29)),(0,0,.8),(0,0,0,1),bodies=(body(),)) for i in range(-10,11))
                if step==0:
                    adapter.reset(state,refs,reset);err('reset_history',adapter.last,last[0]);refs=(refs[10],)*10+refs[10:]
                if step>=35:refs=refs[:12]+(refs[11],)*9
                q=tensor(np.asarray(state.joint_position_rad)[adapter.order]);qd=tensor(np.asarray(state.joint_velocity_rad_s)[adapter.order])
                inv=scope['quat_inv'](tensor(rotation.as_quat()[[3,0,1,2]]))
                data=NS(projected_gravity_b=scope['quat_apply'](inv,tensor([0,0,-1])),root_ang_vel_b=scope['quat_apply'](inv,tensor(state.root_angular_velocity_world_rad_s)),joint_pos=q,joint_vel=qd,default_joint_pos=tensor(adapter.default),default_joint_vel=torch.zeros(1,29))
                env=NS(scene={'robot':NS(data=data)},action_manager=NS(action=last))
                current=scope['proprio'](env);history.append(current);hist=history.buffer
                cmd=scope['MultiMotionCommandV2']();cmd.device='cpu';cmd.motion_ids=torch.zeros(1,dtype=torch.long);cmd.frame_ids=torch.tensor([step]);cmd.motion_anchor_body_index=0
                qref=tensor(np.asarray([r.joint_position_rad for r in refs])[:,adapter.order])
                rots=tensor(np.asarray([r.bodies[0].orientation_xyzw for r in refs])[:,[3,0,1,2]])[:,:,None]
                vel=tensor([r.bodies[0].linear_velocity_world_m_s for r in refs])[:,:,None];omega=tensor([r.bodies[0].angular_velocity_world_rad_s for r in refs])[:,:,None]
                cmd.data_store=NS(gather_centered_window=lambda *a,**k:(qref,None,None,rots,vel,omega))
                command=cmd.get_command_window(torch.tensor([0]));expected_inputs={'proprio':current,'proprio_history':hist,'command_window':command}
                actual=adapter.observations(state,refs)
                for key,a,b in zip(('proprio','history','command'),actual,(current[0],hist[0],command[0])):err(key,a,b)
                native_action=native.act_inference(expected_inputs)
                err('network',adapter.infer(tuple(t[0].numpy() for t in (current,hist,command))),native_action[0])
                term=NS(_offset=torch.zeros(1,29));cmd._env=NS(action_manager=NS(get_term=lambda name:term));cmd.cfg=NS(update_action_offset_with_ref=True);cmd.joint_pos=qref[:,10]
                cmd._maybe_update_action_offset(slice(None));expected_target=(term._offset+native_action)[0,adapter.inverse]
                obs_method=adapter.observations;adapter.observations=lambda *_:actual;output=adapter.act(state,refs);adapter.observations=obs_method
                err('action',adapter.last,native_action[0]);err('motor_target',output.joint_position_target_rad,expected_target)
                last=native_action.clone()
    limits=dict(proprio=2e-5,history=2e-5,command=2e-5,network=0.,action=2e-4,motor_target=2e-4,reset_history=0.)
    report={'passed':all(errors[k]<=limits[k] for k in errors),'cases':160,'resets':4,'max_abs_error':errors,'limits':limits,'config_checks':checks,'checkpoint_sha256':runtime['weights_sha256'],'source_commit':pin['pmt_commit'],'scope':'Original PMT proprio/command/transformer/residual offset and Isaac Lab circular history; simulator and registry glue omitted.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
