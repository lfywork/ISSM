import torch
import numpy as np
import pandas as pd
import argparse

from model import ISSM

def load_args():

	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('--model_type', default=1, type=int)
	parser.add_argument('--horizon', default=10, type=int)
	parser.add_argument('--epochs', default=50, type=int)

	args = parser.parse_args()

	return args

def get_data(loc="https://datahub.io/core/bond-yields-us-10y/r/monthly.csv", header=0):
	df = pd.read_csv(loc, header=header)
	df.set_index('Date')
	ts = df.values[:, 1]
	ts = np.array((ts - np.mean(ts)) / np.std(ts), dtype=np.float)

	return ts

def train(args, issm):
	optim = torch.optim.Adam(issm.parameters(), lr=1e-1)

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
				param.data.clamp_(min=0)
			elif name == 'g':
				param.data.clamp_(min=1e-5, max=0.99)

	for name, param in issm.named_parameters():
		print(name, param.data)

def main(args):
	y = get_data()
	issm = ISSM(y, model_type=args.model_type)
	train(args, issm)
	#issm.forward(horizon=args.horizon)


if __name__ == '__main__':
	args = load_args()
	main(args)