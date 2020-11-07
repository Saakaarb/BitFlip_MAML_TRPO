import os
import numpy as np
import pandas as pd
import itertools as it
from operator import itemgetter
from multiprocessing import Pool
from collections import OrderedDict
try: 
	from maml_rl.envs.CarRacing.dynamics import CarState, DELTA_T, TURN_SCALE, PEDAL_SCALE
	from maml_rl.envs.CarRacing.track import track_name
except: 
	from track import track_name
	DELTA_T = 0.02

root = os.path.dirname(os.path.abspath(__file__))
get_ref_file = lambda ref_name: os.path.join(root, "spec", "refs", f"{ref_name}.csv")
get_map_file = lambda ref_name: os.path.join(root, "spec", "point_maps", f"{ref_name}_ref.npz")

class RefDriver():
	def __init__(self, track_name=track_name, dt=0.01):
		self.ref_name = track_name
		self.ref, self.ref_dt = load_ref(self.ref_name)
		self.load_point_map(self.ref_name, dt=dt)
		self.min_point = np.array([self.Xmap[0], self.Ymap[0]])
		self.max_point = np.array([self.Xmap[-1], self.Ymap[-1]])
		self.start_vel = self.ref.Vx.values[0,0]
		self.start_pos = (self.ref.PathX.values[0,0], self.ref.PathY.values[0,0], self.ref.PathHeading.values[0,0])
		self.max_time = min(self.ref.Time.values[-1,0], np.max(self.time_map))
		self.state_size = self.state(0.0).observation().shape

	def state(self, time):
		index = (time % self.max_time) / self.ref_dt
		index = index.astype(np.int32) if isinstance(index, np.ndarray) else int(index)
		X = self.ref.PathX.values[index][...,0]
		Y = self.ref.PathY.values[index][...,0]
		ψ = self.ref.PathHeading.values[index][...,0]
		S = self.ref.Dist.values[index][...,0]/1000.0
		Vx = self.ref.Vx.values[index][...,0]							#Longitudinal Speed, m/s
		Vy = self.ref.Vy.values[index][...,0]							#Lateral Speed, m/s
		ψ̇ = self.ref.YawVel.values[index][...,0]						#Yaw rate, rad/s
		β = self.ref.SideSlip.values[index][...,0]*np.pi/180			#SideSlip, rad
		δ = self.ref.Steer.values[index][...,0]*np.pi/180				#Steer, rad
		αf = self.ref.SlipAngleFront.values[index][...,0]*np.pi/180		#Front tire slip angle, rad
		αr = self.ref.SlipAngleRear.values[index][...,0]*np.pi/180		#Rear tire slip anlge, rad
		Fxf = self.ref.LongForceFront.values[index][...,0]/1000.0		#Front tire long force, N
		Fxr = self.ref.LongForceRear.values[index][...,0]/1000.0		#Rear tire long force, N
		Fyf = self.ref.LatForceFront.values[index][...,0]/1000.0		#Front tire lat force, N
		Fyr = self.ref.LatForceRear.values[index][...,0]/1000.0			#Rear tire lat force, N
		Fzf = self.ref.VertForceFront.values[index][...,0]/1000.0		#Front tire vert force, N
		Fzr = self.ref.VertForceRear.values[index][...,0]/1000.0		#Rear tire vert force, N
		curvature = self.ref.Curv.values[index][...,0]					#Track curvature, 1/m
		throttle = self.ref.Throttle.values[index][...,0]/100.0			#Throttle, %
		brake = -1*self.ref.Brake.values[index][...,0]/2000.0			#Brake, %
		pedals = (throttle + brake)
		info = OrderedDict(pedals=pedals, curv=curvature, time=self.ref.Time.values[index][...,0])
		state = CarState(X,Y,ψ,Vx,Vy,ψ̇,β,δ,pedals,αf,αr,Fxf,Fxr,Fyf,Fyr,Fzf,Fzr,S,info)
		return state

	def action(self, time, dt):
		states = self.state(time)
		prevstates = self.state(time - dt)
		turn_rate = (states.δ - prevstates.δ)/(TURN_SCALE*dt)
		pedal_rate = (states.pedals - prevstates.pedals)/(PEDAL_SCALE*dt)
		action = np.stack([turn_rate, pedal_rate], -1)
		return action

	def get_sequence(self, time, n, dt=DELTA_T):
		times = (np.array(time).reshape(-1,1) + np.arange(0, n)[None])*dt
		state_spec = self.state(times)
		state = state_spec.observation()
		return np.concatenate([state, np.expand_dims(times,-1)], -1)

	def get_time(self, point, s=0):
		point = np.array(point)
		shape = list(point.shape)
		minref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		maxref = self.max_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		point = np.clip(point, minref, maxref)
		dist = np.clip(s, self.Smap[0], self.Smap[-1])
		dist_ind = np.round((dist-self.Smap[0])/self.sres).astype(np.int32)
		index = np.round((point-minref)/self.res).astype(np.int32)
		times = self.time_map[index[...,0],index[...,1],dist_ind]
		return times

	def load_point_map(self, ref_name, res=1, buffer=50, nthreads=1, ngroups=30, dt=0.01, smaps=5):
		map_file = get_map_file(ref_name)
		if not os.path.exists(map_file):
			self.dt = max(dt, self.ref_dt)
			stride = int(np.round(self.dt/self.ref_dt))
			x = self.ref.PathX.values[::stride]
			y = self.ref.PathY.values[::stride]
			s = self.ref.Dist.values[::stride]/1000.0 if hasattr(self.ref, "Dist") else np.zeros_like(x)
			self.path = np.concatenate([x,y,s], axis=-1)
			x_min, y_min, s_min = map(np.min, [x,y,s])
			x_max, y_max, s_max = map(np.max, [x,y,s])
			X = np.arange(x_min-buffer, x_max+buffer, res).astype(np.float32)
			Y = np.arange(y_min-buffer, y_max+buffer, res).astype(np.float32)
			S = np.linspace(s_min-0.001, s_max+0.001, smaps).astype(np.float32) if smaps>1 and s_min!=s_max else [0]
			points = np.array(list(it.product(X, Y, S)))
			groups = np.split(points, np.arange(0,len(points),len(points)//(ngroups*smaps))[1:])
			if nthreads > 1:
				with Pool(nthreads) as p: times = p.map(self.nearest_point, groups)
			else:
				times = [self.nearest_point(group) for group in groups]
			time = np.concatenate(times, 0)
			time = time.reshape(len(X), len(Y), len(S))
			os.makedirs(os.path.dirname(map_file), exist_ok=True)
			np.savez(map_file, X=X, Y=Y, S=S, time=time, res=res, sres=S[1]-S[0] if len(S)>1 else 1, buffer=buffer, dt=self.dt)
		data = np.load(map_file)
		self.Xmap = data["X"]
		self.Ymap = data["Y"]
		self.Smap = data["S"]
		self.time_map = data["time"]
		self.sres = data["sres"]
		self.res = data["res"]
		self.dt = data["dt"]

	def nearest_point(self, point):
		print(f"Computing {point.shape[0]}")
		points = np.expand_dims(np.array(point),1)
		path = np.expand_dims(self.path, 0)
		dists = np.linalg.norm(path-points, axis=-1)
		idx = np.argmin(dists, -1)
		time = idx * self.dt
		return time

	def __len__(self):
		return len(self.path)

def load_ref(ref_name):
	ref_file = get_ref_file(ref_name)
	df = pd.read_csv(ref_file, header=[0,1], dtype=np.float32)
	ref_dt = np.diff(df.Time.values[0:2,0])[0]
	return df, ref_dt

if __name__ == "__main__":
	ref = RefDriver(ref_name)