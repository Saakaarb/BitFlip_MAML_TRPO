import os
import torch
import numpy as np
import itertools as it
from multiprocessing import Pool
from maml_rl.envs.CarRacing.ref import RefDriver
from maml_rl.envs.CarRacing.dynamics import CarState, DELTA_T
try: from maml_rl.envs.CarRacing.track import Track
except: from track import Track

class CostModel():
	def __init__(self, track=None, ref=None, max_time=np.inf, delta_t=DELTA_T):
		self.track = Track() if track is None else track
		self.X, self.Y = self.track.Xmap, self.track.Ymap
		self.ref = RefDriver(self.track.track_name) if ref is None else ref
		self.max_time = max_time if ref is None else ref.max_time
		self.vmin, self.vmax = 25, 70
		self.delta_t = delta_t

	def __call__(self, action, state, next_state):
		cost = self.get_cost(next_state, state)
		return -cost

	def get_cost(self, state, prevstate=None, times=None, temp=True):
		if prevstate is None: prevstate = state
		if times is not None: times = times*self.delta_t
		cost = self.get_temporal_cost(state, prevstate) if temp else self.get_ref_cost(state, times)
		return cost

	def get_temporal_cost(self, state, prevstate):
		prevpos = np.stack([prevstate.X, prevstate.Y], -1)
		pos = np.stack([state.X, state.Y], -1)
		progress = self.track.get_progress(prevpos, pos)
		dist = self.get_point_cost(pos, transform=False)
		# reward = 2*progress - np.tanh(dist/20)**2 + np.tanh(state.Vx/self.vmin) - np.power(self.vmin-state.Vx,2)/self.vmin**2
		# reward = 2*progress - (dist/20)**1.5 - np.logical_or(state.Vx<self.vmin, state.Vx>self.vmax) - 5*(state.Vx>self.vmin)*np.abs(state.δ) + 0.1*np.tanh(state.Vx/self.vmax)
		reward = (2*progress + 9-(dist/10)**2 - np.logical_or(state.Vx<self.vmin, state.Vx>self.vmax) + 1-(state.Vx/self.vmin)*np.abs(state.δ))/10
		return -reward

	def get_point_cost(self, pos, transform=True):
		idx, dist = self.track.get_nearest(pos) 
		return np.tanh(dist/20)**2 if transform else dist

	def get_ref_cost(self, state, realtime=None):
		if realtime is None: realtime = state.realtime
		realtime = realtime % self.ref.max_time
		pos = np.stack([state.X, state.Y], -1)
		yaw = state.ψ
		vel = state.Vx
		reftime = self.ref.get_time(pos, state.S)
		refposstate = self.ref.state(reftime)
		refpos = np.stack([refposstate.X, refposstate.Y], -1)
		refyaw = refposstate.ψ
		refvel = refposstate.Vx
		timediff = np.abs(realtime - reftime)
		nearest_timediff = np.minimum(timediff, np.abs(timediff - self.ref.max_time))
		cost = 0.1*np.sum((refpos-pos)**2, axis=-1) + (refvel-vel)**2 + (refyaw-yaw)**2 + 50*nearest_timediff**2
		return cost