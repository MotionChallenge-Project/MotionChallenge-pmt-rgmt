"""PMT RGMT: 10-frame history, centered 21-frame command, residual joint targets."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import vector
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file
from network import RGMTActor


class PMTRGMT:
    def __init__(self, root, runtime):
        config = root / runtime['inference_config']
        weights = root / runtime['weights']
        if sha256_file(config) != runtime['inference_config_sha256']:
            raise ValueError('config checksum mismatch')
        if sha256_file(weights) != runtime['weights_sha256']:
            raise ValueError('checkpoint checksum mismatch')
        self.cfg = json.loads(config.read_text())
        self.names = self.cfg['policy_joint_names']
        self.default = vector(self.cfg['default_joint_pos'], self.names).astype(np.float32)
        self.kp = vector(self.cfg['joint_kp'], self.names)
        self.kd = vector(self.cfg['joint_kd'], self.names)
        torch.set_num_threads(1)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        signature = checkpoint['policy_metadata']['signature']
        for key, expected in {'actor_obs_dim': 93, 'history_len': 10, 'cmd_len': 21,
                              'use_vel_estimator': False, 'use_anchor_estimator': False}.items():
            if signature[key] != expected:
                raise ValueError('checkpoint signature mismatch: ' + key)
        self.actor = RGMTActor(checkpoint['model_state_dict'])

    def reset(self, state, reference, seed):
        self.order = [state.joint_names.index(n) for n in self.names]
        self.inverse = [self.names.index(n) for n in state.joint_names]
        self.last = np.zeros(29, np.float32)
        self.history = None

    def observations(self, state, reference):
        if len(reference) != 21:
            raise ValueError('RGMT requires offsets -10 through +10')
        inv = R.from_quat(state.root_orientation_xyzw).inv()
        current = np.concatenate([inv.apply([0, 0, -1]),
            inv.apply(state.root_angular_velocity_world_rad_s),
            np.asarray(state.joint_position_rad)[self.order] - self.default,
            np.asarray(state.joint_velocity_rad_s)[self.order], self.last]).astype(np.float32)
        # Isaac Lab fills a reset history with its first measurement, oldest first.
        if self.history is None:
            self.history = np.repeat(current[None], 10, axis=0)
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = current
        anchors = [next(b for b in f.bodies if b.name == self.cfg['anchor_body']) for f in reference]
        inv_ref = R.from_quat([b.orientation_xyzw for b in anchors]).inv()
        command = np.concatenate([inv_ref.apply([b.linear_velocity_world_m_s for b in anchors]),
            inv_ref.apply([b.angular_velocity_world_rad_s for b in anchors]),
            inv_ref.apply(np.tile([0, 0, -1], (21, 1))),
            np.asarray([f.joint_position_rad for f in reference])[:, self.order]], axis=-1).astype(np.float32)
        if not all(np.isfinite(x).all() for x in (current, self.history, command)):
            raise ValueError('nonfinite RGMT observation')
        return current, self.history.copy(), command

    @torch.inference_mode()
    def infer(self, observations):
        return self.actor(*(torch.from_numpy(x).unsqueeze(0) for x in observations))[0].numpy()

    def act(self, state, reference):
        action = self.infer(self.observations(state, reference))
        if not np.isfinite(action).all():
            raise ValueError('nonfinite RGMT action')
        self.last = action.copy()
        # Author RGMT uses scale=1 and updates action offset to the current reference.
        target = np.asarray(reference[10].joint_position_rad, np.float32)[self.order] + action
        def ordered(x):
            return tuple(float(v) for v in x[self.inverse])
        return ControlCommand(ordered(target), (0.,) * 29, ordered(self.kp), ordered(self.kd), (0.,) * 29)

    def close(self):
        self.actor = None


def create(root, runtime):
    return PMTRGMT(root, runtime)
