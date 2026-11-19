
import sys,types,json
from pathlib import Path
import numpy as np,torch
import argparse
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument("--prepared",required=True,type=Path);parser.add_argument("--weights",required=True,type=Path);parser.add_argument("--device",default="cuda:0")
args=parser.parse_args()
sys.path.insert(0,str(root))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(1)
from tensor_adapter import TensorRGMT
sys.modules.setdefault('onnxruntime',types.ModuleType('onnxruntime'))
from adapter import PMTRGMT
from motion_challenge.simulation.physics import World
from motion_challenge.data.reference import Motion
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket
from safetensors.torch import load_file
from network import RGMTActor
prepared=args.prepared
from motion_challenge.resources import resource_root
meta=json.loads((prepared/'dataset.json').read_text());poses=np.load(prepared/'poses.npy',mmap_mode='r')
worlds=[World(resource_root()/'robots/g1-29dof-mjwarp-v1.json') for _ in range(7)]
names=worlds[0].joint_names
for clip,w in enumerate(worlds):
 p=poses[meta['starts'][clip]:meta['starts'][clip]+meta['lengths'][clip]]
 w.reset(Motion(names,tuple(np.arange(len(p))/30.),p[:,7:],p[:,:3],p[:,3:7]))
bank=TensorMotionBank(np.array(poses),meta['starts'],meta['lengths'],np.full(len(meta['names']),meta['fps']),meta['names'],args.device)
refs=TensorReference(worlds[0],bank,147,args.device)
runtime=json.loads((root/'runtime.mjwarp.json').read_text())
adapter=TensorRGMT(root,runtime,names,7,args.device,args.weights)
original=[]
for _ in range(7):
 o=PMTRGMT.__new__(PMTRGMT);o.cfg=json.loads((root/'policy.json').read_text());o.names=o.cfg['policy_joint_names'];o.default=np.asarray(o.cfg['default_joint_pos'],np.float32);original.append(o)
oracle=RGMTActor(load_file(str(args.weights/'actor.safetensors'))).eval()
rng=np.random.default_rng(1771);maximum=0.;action_error=0.;pd_error=0.
for step in range(20):
 offsets=np.array(runtime['reference_offsets_s']);times=np.arange(7)[:,None]*.03+step*.02+offsets
 reference=refs.motion(torch.arange(7,device=args.device),torch.tensor(times,device=args.device))
 q=rng.normal(0,.2,(7,29));qd=rng.normal(0,.4,(7,29));rotation=rng.normal(size=(7,4));rotation/=np.linalg.norm(rotation,axis=1,keepdims=True);omega=rng.normal(0,.1,(7,3))
 state=DeployableStateBatch(names,torch.full((7,),step*.02,device=args.device),*[torch.tensor(x,device=args.device) for x in (q,qd,rotation,omega)])
 if step==8:adapter.reset(torch.tensor([1,5],device=args.device))
 actual=adapter.observations(state,reference);expected=[[],[],[]]
 for row,(world,old) in enumerate(zip(worlds,original)):
  packet=DeployableStatePacket(step*.02,names,tuple(q[row]),tuple(qd[row]),tuple(rotation[row]),tuple(omega[row]))
  frames=world.references(times[row])
  if step==0 or (step==8 and row in (1,5)):old.reset(packet,frames,0)
  for j,x in enumerate(old.observations(packet,frames)):expected[j].append(x)
 for j in range(3):
  e=np.stack(expected[j]);a=actual[j].cpu().numpy();maximum=max(maximum,float(np.max(abs(e-a))))
  np.testing.assert_allclose(a,e,atol=2e-6,rtol=2e-6)
 with torch.inference_mode():
  wanted=torch.cat([oracle(*(torch.tensor(np.asarray(expected[j][i]))[None] for j in range(3))) for i in range(7)])
  result=adapter.actor(*actual).cpu()
 error=float((result-wanted).abs().max());action_error=max(action_error,error)
 np.testing.assert_allclose(result.numpy(),wanted.numpy(),atol=2e-5,rtol=2e-5)
 adapter.last.copy_(wanted.to(args.device))
 for row,old in enumerate(original):old.last=wanted[row].numpy().copy()
report={'status':'passed','steps':20,'batch':7,'partial_reset_slots':[1,5],'observation_max_abs_error':maximum,'network_max_abs_error':action_error,'reference_fk_oracle':'CPU MuJoCo native kinematics/Jacobians with independent interpolation'}
(root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'validation/mjwarp-network-parity.json').write_text((args.weights/'parity.json').read_text())
runtime['tensor_entrypoint']='tensor_adapter:TensorRGMT';(root/'runtime.mjwarp.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(report),flush=True)

