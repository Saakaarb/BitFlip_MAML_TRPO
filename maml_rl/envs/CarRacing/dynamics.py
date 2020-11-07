import os
import sys
import gym
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
from maml_rl.envs.CarRacing.config import Config

DELTA_T = 0.02
TURN_SCALE = 0.2
PEDAL_SCALE = 20.0
TURN_LIMIT = np.pi
PEDAL_LIMIT = 1.0

constants = Config(
	m = 1370.0, 			# Mass (kg)
	I_zz = 4453.0, 			# Inertia (kg m^2)
	l_f = 1.293, 			# Distance from CG to front axle (m)
	l_r = 1.475, 			# Distance from CG to rear axle (m)
	δ_ratio = 17.85, 		# Steer ratio
	F_ZFStatic = 3577.0, 	# F Static tire normal force (N)
	F_ZRStatic = 3136.0, 	# R Static tire normal force (N)
	ρ = 1.205, 				# Air density (kg/m^3)
	SA = 2.229, 			# Surface area (m^2)
	C_LF = 0.392, 			# Coefficient of front down force
	C_LR = 0.918, 			# Coefficient of rear down force
	C_D = 0.6, 				# Coefficient of drag
	C_αf = 312631.0, 		# Front tire cornering stiffness (N/rad)
	C_αr = 219079.0, 		# Rear tire cornering stiffness (N/rad)
	μ_f = 1.612, 			# Front tire friction
	μ_r = 1.587, 			# Rear tire friction
	Mz = 0,					# Tire aligning torque (Nm)
	F_RR = 0,				# Tire rolling resistance force (N)
	F_YAero = 0,			# Aero side force (N)
	M_ZAero = 0,			# Aero yaw moment (Nm)
)

def calc_F_Aero(Vx,Vy,C): 
	return (0.5*constants.ρ) * (Vx**2 + Vy**2) * (constants.SA*C)

def calc_Fz(F_ZStatic, F_Z_Aero): 
	return F_ZStatic + 0.5*F_Z_Aero

def calc_Fy(α, μ, C_α, F_Z, tan=np.tan):
	return np.where(np.abs(α) < 3*μ*F_Z/C_α, C_α*tan(α) - C_α**2/(3*μ*F_Z)*np.abs(tan(α))*tan(α) + C_α**3/(27*(μ*F_Z)**2)*(tan(α)**3), μ*F_Z*np.sign(α))

def calc_Fx(μ, F_Y, F_Z): 
	return F_Z * np.abs(np.sqrt(np.maximum(μ**2 - (F_Y / np.maximum(F_Z,1e-8))**2, 1e-8)))

def clamp(x, r):
	return np.clip(x, -r, r)

class CarState():
	def __init__(self, *args, **kwargs):
		self.update(*args, **kwargs)

	def update(self,X=None,Y=None,ψ=None,Vx=None,Vy=None,ψ̇=None,β=None,δ=None,pedals=None,αf=None,αr=None,FxF=None,FxR=None,FyF=None,FyR=None,FzF=None,FzR=None,S=None,info={}):
		givens = [x for x in [X,Y,ψ,Vx,Vy,ψ̇,β,δ,pedals,αf,αr,FxF,FxR,FyF,FyR,FzF,FzR,S] if x is not None]
		default = lambda: givens[0]*0 if len(givens) > 0 else 0.0
		self.X = X if X is not None else default()
		self.Y = Y if Y is not None else default()
		self.ψ = ψ if ψ is not None else default()
		self.Vx = Vx if Vx is not None else default()
		self.Vy = Vy if Vy is not None else default()
		self.ψ̇  = ψ̇  if ψ̇  is not None else default()
		self.β = β if β is not None else default()
		self.δ = δ if δ is not None else default()
		self.pedals = pedals if pedals is not None else default()
		self.αf = αf if αf is not None else default()
		self.αr = αr if αr is not None else default()
		self.FxF = FxF if FxF is not None else default()
		self.FxR = FxR if FxR is not None else default()
		self.FyF = FyF if FyF is not None else default()
		self.FyR = FyR if FyR is not None else default()
		self.FzF = FzF if FzF is not None else default()
		self.FzR = FzR if FzR is not None else default()
		self.S = S if S is not None else default()
		self.info = info
		self.shape = getattr(default(), "shape", ())
		return self

	def observation(self):
		pos_x = np.expand_dims(self.X, axis=-1)
		pos_y = np.expand_dims(self.Y, axis=-1)
		rot_f = np.expand_dims(self.ψ, axis=-1)
		vel_f = np.expand_dims(self.Vx, axis=-1)
		vel_s = np.expand_dims(self.Vy, axis=-1)
		yaw_dot = np.expand_dims(self.ψ̇, axis=-1)
		beta = np.expand_dims(self.β, axis=-1)
		steer = np.expand_dims(self.δ, axis=-1)
		pedals = np.expand_dims(self.pedals, axis=-1)
		αf = np.expand_dims(self.αf, axis=-1)
		αr = np.expand_dims(self.αr, axis=-1)
		FxF = np.expand_dims(self.FxF, axis=-1)
		FxR = np.expand_dims(self.FxR, axis=-1)
		FyF = np.expand_dims(self.FyF, axis=-1)
		FyR = np.expand_dims(self.FyR, axis=-1)
		FzF = np.expand_dims(self.FzF, axis=-1)
		FzR = np.expand_dims(self.FzR, axis=-1)
		S = np.expand_dims(self.S, axis=-1)
		return np.concatenate([pos_x, pos_y, rot_f, vel_f, vel_s, yaw_dot, beta, steer, pedals, αf, αf, FxF, FxR, FyF, FyR, FzF, FzR, S], axis=-1)

	@staticmethod
	def observation_spec(state):
		pos_x = state[...,0]
		pos_y = state[...,1]
		rot_f = state[...,2]
		vel_f = state[...,3]
		vel_s = state[...,4]
		yaw_dot = state[...,5]
		beta = state[...,6]
		steer = state[...,7]
		pedals = state[...,8]
		αf = state[...,9]
		αr = state[...,10]
		FxF = state[...,11]
		FxR = state[...,12]
		FyF = state[...,13]
		FyR = state[...,14]
		FzF = state[...,15]
		FzR =  state[...,16]
		S =  state[...,17]
		state_spec = CarState(pos_x,pos_y,rot_f,vel_f,vel_s,yaw_dot,beta,steer,pedals,αf,αr,FxF,FxR,FyF,FyR,FzF,FzR,S,{})
		return state_spec

	def print(self):
		return f"X: {self.X:4.3f}, Y: {self.Y:4.3f}, ψ: {self.ψ:4.3f}, Vx: {self.Vx:4.3f}, Vy: {self.Vy:4.3f}, δ: {self.δ:4.3f}, pedals: {self.pedals:4.3f}, Fx: ({self.FxF:4.3f}, {self.FxR:4.3f}), Fy: ({self.FyF:4.3f}, {self.FyR:4.3f}), Fz: ({self.FzF:4.3f}, {self.FzR:4.3f})"

class CarDynamics():
	def __init__(self, *kwargs):
		self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))

	def reset(self, start_pos, start_vel, turn_adjust=0.0, pedal_adjust=0.0):
		self.state = CarState(X=start_pos[0], Y=start_pos[1], ψ=start_pos[2], Vx=start_vel)
		self.turn_scale = TURN_SCALE * 2**(turn_adjust)
		self.pedal_scale = PEDAL_SCALE * (1+pedal_adjust)
		self.turn_limit = TURN_LIMIT * 2**(-2-turn_adjust)
		self.pedal_limit = PEDAL_LIMIT * (1-0.5*pedal_adjust)

	def step(self, action, dt=DELTA_T, integration_steps=1, use_delta=False):
		turn_rate = action[...,0]
		pedal_rate = action[...,1]
		dt = dt/integration_steps
		state = self.state
		
		for i in range(integration_steps):
			F_ZF_Aero = calc_F_Aero(state.Vx, state.Vy, constants.C_LF)
			F_ZR_Aero = calc_F_Aero(state.Vx, state.Vy, constants.C_LR)
			F_X_Aero = calc_F_Aero(state.Vx, state.Vy, constants.C_D)
			Fy_scale = np.minimum(np.abs(state.Vx), 1)

			δ = clamp(state.δ + self.turn_scale*turn_rate * dt, TURN_LIMIT) if use_delta else self.turn_limit*turn_rate
			αf = -np.arctan2((state.Vy + constants.l_f * state.ψ̇),state.Vx) + δ
			αr = -np.arctan2((state.Vy - constants.l_r * state.ψ̇),state.Vx) + 0.0
			
			pedals = clamp(state.pedals + self.pedal_scale*pedal_rate * dt, PEDAL_LIMIT) if use_delta else self.pedal_limit*pedal_rate
			acc = np.maximum(pedals, 0)
			accel = np.maximum(-(acc**3)*10523.0 + (acc**2)*12394.0 + (acc)*1920.0, 0)
			brake = np.minimum(pedals, 0)*22500*(self.state.Vx > 0)

			FzF = calc_Fz(constants.F_ZFStatic, F_ZF_Aero)
			FzR = calc_Fz(constants.F_ZRStatic, F_ZR_Aero)
			FyF = calc_Fy(αf, constants.μ_f, constants.C_αf, FzF) * Fy_scale
			FyR = calc_Fy(αr, constants.μ_r, constants.C_αr, FzR) * Fy_scale
			FxF = clamp(brake*0.6, calc_Fx(constants.μ_f, FyF, FzF))
			FxR = clamp(accel+brake*0.4, calc_Fx(constants.μ_r, FyR, FzR))
			
			ψ̈ = (1/constants.I_zz) * ((2*FxF * np.sin(δ) + 2*FyF * np.cos(δ)) * constants.l_f - 2*FyR * constants.l_r)
			V̇x = (1/constants.m) * (2*FxF * np.cos(δ) - 2*FyF * np.sin(δ) + 2*FxR - F_X_Aero) + state.ψ̇ * state.Vy
			V̇y = (1/constants.m) * (2*FyF * np.cos(δ) + 2*FxF * np.sin(δ) + 2*FyR) - state.ψ̇ * state.Vx
			
			ψ̇ = state.ψ̇ + ψ̈  * dt
			Vx = state.Vx + V̇x * dt
			Vy = state.Vy + V̇y * dt
			
			β = np.arctan2(Vy,Vx)
			ψ = (state.ψ + ψ̇  * dt)
			X = state.X + (Vx * np.cos(ψ) - Vy * np.sin(ψ)) * dt
			Y = state.Y + (Vy * np.cos(ψ) + Vx * np.sin(ψ)) * dt
			S = (state.S*1000 + np.sqrt(Vx**2 + Vy**2) * dt)/1000

			info = {"F_ZF_Aero":F_ZF_Aero, "F_ZR_Aero":F_ZR_Aero, "F_X_Aero":F_X_Aero, "yaw_acc":ψ̈ , "vx_dot":V̇x, "vy_dot":V̇y}
			state = state.update(X,Y,ψ,Vx,Vy,ψ̇,β,δ,pedals,αf,αr,FxF/1000,FxR/1000,FyF/1000,FyR/1000,FzF/1000,FzR/1000,S,info)

		self.state = state

	def observation(self, state):
		state = self.state if state == None else state
		return state.observation()

	@staticmethod
	def observation_spec(state, device=None):
		return CarState.observation_spec(state)

	def set_state(self, state, device=None):
		self.state = self.observation_spec(state, device=device)