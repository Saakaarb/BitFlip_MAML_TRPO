import os
import gym
import argparse
import numpy as np
import keyboard as kbd
from maml_rl.envs.CarRacing.config import Config
np.set_printoptions(precision=3, sign=" ")
	
class InputController():
	def __init__(self, state_size, action_size, config=None, **kwargs):
		self.state_size = state_size
		self.action_size = action_size

	def get_action(self, state, eps=0.0, sample=False):
		shape = state.shape[:-len(self.state_size)]
		action_size = [*shape, *self.action_size]
		action = np.zeros(action_size) if not sample else np.random.random(action_size)
		try:
			if kbd.is_pressed("left"):
				action[...,0] += 1
			if kbd.is_pressed("right"):
				action[...,0] -= 1
			if kbd.is_pressed(kbd.KEY_UP):
				action[...,1] += 1
			if kbd.is_pressed(kbd.KEY_DOWN):
				action[...,1] -= 1
		except Exception as e:
			print(e)
		return action

def test_input(nsteps=400):
	env = gym.make("CarRacing-cubic-v1")
	agent = InputController(env.observation_space.shape, env.action_space.shape)
	state = env.reset()
	total_reward = None
	done = False
	for step in range(0,nsteps):
		action = agent.get_action(state, 0.0, False)
		state, reward, done, info = env.step(action)
		total_reward = reward  if total_reward is None else total_reward + reward
		log_string = f"Step: {step:8d}, Reward: {reward:5.3f}, Action: {np.array2string(action, separator=',')}, Done: {done}"
		print(log_string)
		env.render()
	print(f"Reward: {total_reward}")
	env.close()

if __name__ == "__main__":
	test_input()