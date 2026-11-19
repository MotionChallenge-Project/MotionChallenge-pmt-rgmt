"""PMT RGMT batched deployment using safe actor weights."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from network import RGMTActor
from motion_challenge.validation import sha256_file
from motion_challenge.simulation.tensor_math import normalize, conjugate, multiply, rotate


def axis_rotation(angle, axis):
    xyz = [torch.zeros_like(angle) for _ in range(3)]
    xyz[axis] = (angle*.5).sin()
    return torch.stack((*xyz, (angle*.5).cos()), dim=-1)


class TensorRGMT:
    def __init__(self, root, runtime, joint_names, slots, device, converted_weights):
        self.device = torch.device(device)
        self.cfg = json.loads((Path(root)/"policy.json").read_text())
        if self.cfg["anchor_body"] != "torso_link": raise ValueError("RGMT requires the G1 torso anchor")
        self.names = self.cfg["policy_joint_names"]
        self.order = [joint_names.index(n) for n in self.names]
        self.inverse = [self.names.index(n) for n in joint_names]
        self.waist = [joint_names.index("waist_"+axis+"_joint") for axis in ("yaw","roll","pitch")]
        self.default = torch.tensor(self.cfg["default_joint_pos"], dtype=torch.float32, device=device)
        self.kp = torch.tensor(self.cfg["joint_kp"], dtype=torch.float32, device=device)[self.inverse]
        self.kd = torch.tensor(self.cfg["joint_kd"], dtype=torch.float32, device=device)[self.inverse]
        folder = Path(converted_weights)
        meta = json.loads((folder/"conversion.json").read_text())
        if meta["source_sha256"] != runtime["weights_sha256"]: raise ValueError("Source checksum mismatch")
        if sha256_file(folder/"actor.safetensors") != meta["weights_sha256"]: raise ValueError("Converted checksum mismatch")
        self.actor = RGMTActor(load_file(str(folder/"actor.safetensors"))).to(device).eval()
        self.history = torch.zeros((slots,10,93), device=device)
        self.last = torch.zeros((slots,29), device=device)
        self.fresh = torch.ones(slots, device=device, dtype=torch.bool)

    def reset(self, slots):
        self.last[slots] = 0
        self.fresh[slots] = True

    def torso(self, reference):
        root = normalize(reference.root_orientation_xyzw.double())
        q = reference.joint_position_rad.double()
        qd = reference.joint_velocity_rad_s.double()
        omega = reference.root_angular_velocity_world_rad_s.double()
        linear = reference.root_linear_velocity_world_m_s.double()
        yaw = multiply(root, axis_rotation(q[...,self.waist[0]],2))
        axis_z = torch.tensor([0.,0.,1.],device=self.device,dtype=torch.float64)
        angular = omega + rotate(root,axis_z)*qd[...,self.waist[0],None]
        translation = torch.tensor([-.0039635,0.,.044],device=self.device,dtype=torch.float64)
        offset = rotate(yaw,translation)
        linear = linear + torch.linalg.cross(angular,offset)
        rotation = yaw
        for index,axis in zip(self.waist[1:],(0,1)):
            vector = torch.zeros(3,device=self.device,dtype=torch.float64);vector[axis]=1
            angular = angular + rotate(rotation,vector)*qd[...,index,None]
            rotation = multiply(rotation,axis_rotation(q[...,index],axis))
        return normalize(rotation),linear,angular

    def observations(self,state,reference):
        inv = conjugate(normalize(state.root_orientation_xyzw.double()))
        gravity = torch.tensor([0.,0.,-1.],device=self.device,dtype=torch.float64)
        q = state.joint_position_rad[:,self.order].double()
        qd = state.joint_velocity_rad_s[:,self.order].double()
        current = torch.cat((rotate(inv,gravity),rotate(inv,state.root_angular_velocity_world_rad_s.double()),
                             q-self.default,qd,self.last),dim=-1).float()
        shifted = torch.cat((self.history[:,1:],current[:,None]),dim=1)
        self.history.copy_(torch.where(self.fresh[:,None,None],current[:,None],shifted))
        self.fresh.zero_()
        rotation,linear,angular = self.torso(reference)
        ri = conjugate(rotation)
        command = torch.cat((rotate(ri,linear),rotate(ri,angular),rotate(ri,gravity),
                             reference.joint_position_rad[:,:,self.order]),dim=-1).float()
        return current,self.history.clone(),command

    @torch.inference_mode()
    def act(self,state,reference):
        action = self.actor(*self.observations(state,reference))
        self.last.copy_(action)
        target = (reference.joint_position_rad[:,10,self.order].float()+action)[:,self.inverse]
        zero = torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1)
