import torch
import numpy as np
import pandas as pd
import argparse

from model import ISSM

def load_args():

	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('--model_type', default=1, type=int)
	parser.add_argument('--horizon', default=10, type=int)

	args = parser.parse_args()

	return args

def get_data(loc="https://datahub.io/core/bond-yields-us-10y/r/monthly.csv", header=0):
	df = pd.read_csv(loc, header=header)
	df.set_index('Date')
	ts = df.values[:, 1]
	ts = np.array((ts - np.mean(ts)) / np.std(ts), dtype=np.float)

	return ts


def main(args):
	y = get_data()
	issm = ISSM(y, model_type=args.model_type)
	issm.forward(horizon=args.horizon)


if __name__ == '__main__':
	args = load_args()
	main(args)