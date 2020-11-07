import os
import gym
import sys
import torch
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
from collections import OrderedDict
from maml_rl.envs.CarRacing.dynamics import CarDynamics, CarState, DELTA_T
from maml_rl.envs.CarRacing.viewer import PathAnimator
from maml_rl.envs.CarRacing.track import Track, track_name
from maml_rl.envs.CarRacing.cost import CostModel
from maml_rl.envs.CarRacing.ref import RefDriver
np.set_printoptions(precision=3, sign=" ")

def register_track(cls, track):
	name = os.path.splitext(track)[0]
	gym.register(f"{cls.name.replace('-',f'-{name}-')}", entry_point=lambda: cls(name))

class EnvMeta(type):
	def __new__(meta, name, bases, class_dict):
		cls = super().__new__(meta, name, bases, class_dict)
		gym.register(cls.name, entry_point=cls)
		for track in os.listdir(os.path.join(os.path.dirname(__file__), "CarRacing", "spec", "tracks")):
			register_track(cls, track)
		return cls

class CarRacingV1(gym.Env, metaclass=EnvMeta):
	name = "CarRacing-v1"
	def __init__(self, track_name=track_name, max_time=None, delta_t=2.5*DELTA_T, withtrack=False):
		self.ref = RefDriver(track_name)
		self.track = Track(track_name)
		self.dynamics = CarDynamics()
		self.delta_t = delta_t
		self.withtrack = withtrack
		self.max_time = self.ref.max_time if max_time is None else max_time
		self.cost_model = CostModel(self.track, self.ref, self.max_time, self.delta_t)
		self.action_space = self.dynamics.action_space
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.reset().shape)
		self.spec = gym.envs.registration.EnvSpec(f"CarRacing-{track_name}-v1", max_episode_steps=int(self.max_time/self.delta_t))
		self.spec.max_episode_steps = int(self.max_time/self.delta_t)

	def reset(self, train=True, meta=True):
		self.time = 0
		self.realtime = 0.0
		self.action = np.zeros(self.action_space.shape)
		self.dynamics.reset(self.ref.start_pos, self.ref.start_vel)
		self.state_spec, state = self.observation()
		self.info = {"ref":{}, "car":{}}
		self.done = False
		return state

	def step(self, action, device=None, info=True):
		self.time += 1
		self.realtime = self.time * self.delta_t
		self.dynamics.step(action, dt=self.delta_t, use_delta=False)
		next_state_spec, next_state = self.observation()
		reward = -self.cost_model.get_cost(next_state_spec, self.state_spec, self.time, True)
		trackdist = self.track.get_nearest(next_state[...,[0,1]])[1]
		done = np.logical_or(trackdist > 40.0, self.done)
		done = np.logical_or(next_state_spec.Vx < 8.0, done)
		self.done = np.logical_or(self.realtime >= self.max_time, done)
		self.info = self.get_info(reward, action) if info else {"ref":{}, "car":{}}
		self.state_spec = next_state_spec
		return next_state, reward, done, self.info

	def render(self, mode="human", **kwargs):
		if not hasattr(self, "viewer"): self.viewer = PathAnimator(self.track, interactive=mode!="video")
		ref_spec = self.ref.state(self.realtime)
		pos = np.stack([self.state_spec.X, self.state_spec.Y], -1)
		refpos = np.stack([ref_spec.X, ref_spec.Y], -1)
		car = np.stack([pos, pos + np.array([np.cos(self.state_spec.ψ ), np.sin(self.state_spec.ψ)])])
		ref = np.stack([refpos, refpos + np.array([np.cos(ref_spec.ψ ), np.sin(ref_spec.ψ )])])
		return self.viewer.animate_path(pos, [car, ref], info=self.info, path=self.track.path if self.withtrack else None, **kwargs)

	def observation(self, carstate=None):
		dyn_state = self.dynamics.observation(carstate)
		realtime = np.expand_dims(self.realtime, axis=-1)
		state = np.concatenate([dyn_state, realtime], axis=-1)
		self.dynamics_size = state.shape[-1]
		spec = self.observation_spec(state)
		dyn_meta = np.concatenate([np.ones_like(realtime)*self.dynamics.turn_scale, np.ones_like(realtime)*self.dynamics.pedal_scale], -1)
		state = np.concatenate([state, self.track.get_path(dyn_state[...,[0,1]], heading=spec.ψ), dyn_meta], -1) if self.withtrack else state
		return spec, state

	@staticmethod
	def observation_spec(observation):
		dyn_state = observation[...,:-1]
		dyn_spec = CarState.observation_spec(dyn_state)
		realtime = observation[...,-1]
		dyn_spec.realtime = realtime
		return dyn_spec

	def set_state(self, state, device=None, times=None):
		if isinstance(state, torch.Tensor): state = state.cpu().numpy()
		dyn_state = state[...,:-1]
		self.dynamics.set_state(dyn_state, device=device)
		self.realtime = state[...,-1] if times is None else times*self.delta_t 
		self.state_spec = self.observation()[0]
		self.time = self.realtime / self.delta_t

	def get_info(self, reward, action):
		dynspec = self.dynamics.state
		refspec = self.ref.state(self.realtime)
		refaction = self.ref.action(self.realtime, self.delta_t)
		reftime = self.ref.get_time(np.stack([dynspec.X, dynspec.Y], -1), dynspec.S)
		carinfo = info_stats(dynspec, reftime, reward, action)
		refinfo = info_stats(refspec, self.realtime, 0, refaction)
		info = {"ref": refinfo, "car": carinfo}
		return info

	def close(self, path=None):
		if hasattr(self, "viewer"): self.viewer.close(path)
		self.closed = True

	def sample_tasks(self, num_tasks):
		tasks=[{'task':np.random.random(2)} for i in range(num_tasks)]
		return tasks

	def reset_task(self, task):
		self.dynamics.reset(self.ref.start_pos, self.ref.start_vel, *task["task"])

def info_stats(state, realtime, reward, action):
	turn_rate = action[...,0]
	pedal_rate = action[...,1]
	info = {
		"Time": f"{realtime:7.2f}",
		"Pos": f"{{'X':{justify(state.X)}, 'Y':{justify(state.Y)}}}",
		"Vel": f"{{'X':{justify(state.Vx)}, 'Y':{justify(state.Vy)}}}",
		"Dist": np.round(state.S, 4),
		"Yaw angle": np.round(state.ψ, 4),
		"Yaw vel": np.round(state.ψ̇, 4),
		"Beta": np.round(state.β, 4),
		"Alpha": f"{{'F':{justify(state.αf)}, 'R':{justify(state.αr)}}}",
		"Fz": f"{{'F':{justify(state.FzF)}, 'R':{justify(state.FzR)}}}",
		"Fy": f"{{'F':{justify(state.FyF)}, 'R':{justify(state.FyR)}}}",
		"Fx": f"{{'F':{justify(state.FxF)}, 'R':{justify(state.FxR)}}}",
		"Steer angle": np.round(state.δ, 4),
		"Pedals": np.round(state.pedals, 4),
		"Reward": np.round(reward, 4),
		"Action": f"{{'Trn':{justify(turn_rate)}, 'ped':{justify(pedal_rate)}}}"
	}
	return info

def justify(num): return str(np.round(num, 3)).rjust(10,' ')
