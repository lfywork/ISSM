import torch
import numpy as np
import pandas as pd
import argparse

from model import ISSM
from utils import plot_forecasts

def load_args():

	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('--model_type', default=3, type=int)
	parser.add_argument('--horizon', default=8, type=int)
	parser.add_argument('--epochs', default=50, type=int)

	args = parser.parse_args()

	return args

def get_data(loc="https://datahub.io/core/bond-yields-us-10y/r/monthly.csv", header=0):
	df = pd.read_csv(loc, header=header)
	df.set_index('Date')
	ts = df.values[:, 1]
	ts = np.array((ts - np.mean(ts)) / np.std(ts), dtype=np.float)

	return ts

def get_elecequip(args):
	elecequip = pd.read_csv('elec.csv')
	elecequip = elecequip.iloc[:, 1]

	dates_index = pd.date_range('1996-01', periods=195, freq='M')
	elecequip.index = dates_index
	elecequip.name = 'elecequip'

	# time-series
	ts = elecequip.values

	# Normalize
	ts = np.array((ts - np.mean(ts)) / np.std(ts), dtype=np.float)

	return ts[:-args.horizon], ts

def train(args, issm):
	optim = torch.optim.Adam(issm.parameters(), lr=1e-2)

	for i in range(args.epochs):
		#def closure():
		optim.zero_grad()
		loss = issm.forward(horizon=args.horizon)
		loss.backward()
		print(loss.detach().item())
		#return loss
		optim.step()

		for name, param in issm.named_parameters():
			if name == 'sigma':
				param.data.clamp_(min=1e-5)
			elif name == 'g':
				latent_dim = param.data.shape[0]
				if latent_dim == 1:
					# clamp alpha
					param.data.clamp_(min=1e-5, max=0.999)
				elif latent_dim >= 2:
					# clamp alpha
					param.data[0].clamp_(min=1e-5, max=0.999)
					# clamp beta
					for j in range(len(param[0])):
						param.data[1][j].clamp_(min=1e-5, max=param[0][j].detach().item())
				if latent_dim > 2:
					# clamp gamma
					for j in range(len(param[0])):
						param.data[2][j].clamp_(min=1e-5, max=1-param[0][j].detach().item())
					param.data[3:].clamp_(min=0., max=0.)
			elif name == 'm_prior':
				latent_dim = param.data.shape[0]
				# if latent_dim == 1:
				# 	# clamp level
				# 	param.data.clamp_(min=-2, max=2)
				# elif latent_dim >= 2:
				# 	# clamp level
				# 	param.data[0].clamp_(min=-2, max=2)
				# 	# clamp trend
				# 	param.data[1].clamp_(min=-2, max=2)
				if latent_dim > 2:
					# clamp seasonality
					#param.data[2:].clamp_(min=-2, max=2)
					param.data[2:] = param.data[2:] - torch.mean(param.data[2:])
				#param.data.clamp_(min=-2, max=2)
			elif name == 'S_prior':
				param.data = param.data * torch.eye(param.data.shape[0])
				param.data.clamp_(min=1e-8)
			elif name == 'b':
				param.data.clamp_(min=-1, max=1)

	# for name, param in issm.named_parameters():
	# 	print(name, param.data)

def main(args):
	#y = get_data()
	y, full_y = get_elecequip(args)
	issm = ISSM(y, model_type=args.model_type)
	train(args, issm)
	with torch.no_grad():
		#print(issm.m_prior)
		#print(torch.sum(issm.m_prior))
		forecasts_mean, forecasts_std = issm.generate(horizon=args.horizon)
		plot_forecasts(forecasts_mean, forecasts_std, y, full_y)
	#issm.forward(horizon=args.horizon)


if __name__ == '__main__':
	args = load_args()
	main(args)