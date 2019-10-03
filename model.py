import torch
import numpy as np
from torch import nn
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
sns.set()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class ISSM(nn.Module):
	def __init__(self, obs, model_type=1):
		super(ISSM, self).__init__()

		self.obs = obs
		self.T = len(obs)
		self.z = torch.tensor(obs).view(self.T, 1).float()
		self.model_type = model_type

		self.init_model()

	def init_model(self):
		# level model
		if self.model_type == 1:
			self.latent_dim = 1
			self.F = torch.ones(1, 1, self.T)  # 1x1xT
			self.a = torch.ones(1, self.T)     # 1xT
			self.g = 0.5 * torch.ones(1, self.T)    # 1xT

		# level/trend model
		elif self.model_type == 2:
			self.latent_dim = 2
			g_t = torch.tensor([0.5, 0.1])  # 2x1
			self.g = g_t.repeat(self.T, 1).view(self.latent_dim, self.T)  # 2xT
			F_t = torch.tensor([1.0, 1.0, 0., 1.0]).view(self.latent_dim, self.latent_dim, 1) # 2x2x1 matrix here
			a_t = torch.tensor([1.0, 1.0]).view(self.latent_dim, 1)  # 2x1
			self.F = F_t.repeat(1, 1, self.T) 						# 2x2xT
			self.a = a_t.repeat(1, self.T) 							# 2xT

		# monthly seasonal model with level/trend
		elif self.model_type == 3:
			self.latent_dim = 13
			g_t = torch.tensor([0.5, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 13x1
			self.g = g_t.repeat(self.T, 1).view(self.latent_dim, self.T)  # 13xT
			F_t = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]).float()
			self.F = F_t.view(self.latent_dim, self.latent_dim, 1).repeat(1, 1, self.T)  # 13x13xT
			a_t = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).float().view(self.latent_dim, 1)  # 13x1
			self.a = a_t.repeat(1, self.T)  # 13xT


		# priors for latent space vector
		# mean and variance for each dimension
		self.m_prior = torch.zeros(self.latent_dim, 1)  # mu_prior = 0
		sigma_prior = 1e2*torch.ones(self.latent_dim)
		self.S_prior = torch.diag(sigma_prior)

		# sigma_t
		# b_t
		# z_t
		self.sigma = 0.5 * torch.ones(self.T, 1).float()
		self.b = torch.zeros(self.T, 1).float()
		self.sigma = torch.nn.Parameter(self.sigma)
		self.g = torch.nn.Parameter(self.g)

	# filtering
	# stolen from https://gluon.mxnet.io/chapter12_time-series/issm-scratch.html#Filtering

	def ISSM_filter(self, z, b, F, a, g, sigma, m_prior, S_prior):
		H = F.shape[0]  # dimension of latent space
		#print('H:', H)
		T = z.shape[0]  # number of observations
		#print('T:', T)
		eye_h = torch.tensor(np.eye(H)).float()
		
		mu_seq = []
		S_seq = []
		log_p_seq = []
		deltas = []
		total_loss = 0
		for t in range(T):
			if t == 0:
				# At first time step, use mu_0, S_0 (prior)
				mu_h = m_prior
				S_hh = S_prior
			
			else:
				# Compute using update equations
				F_t = F[:, :, t]
				g_t = g[:, t].view(H,1)
				
				mu_h = torch.matmul(F_t, mu_t)  # first eqn
				S_hh = torch.matmul(F_t, torch.matmul(S_t, F_t.t())) + torch.matmul(g_t, g_t.t())  # third eqn
				# S_t is from previous time step, not defined for first pass
			#print('mu_h:', mu_h.shape)
			#print('S_hh:', S_hh.shape)
			a_t = a[:, t].view(H,1)
			#print('a_t:', a_t.shape)
			mu_v = torch.matmul(a_t.t(), mu_h)  # second eqn
			#print('mu_v:', mu_v.shape)
			
			# Compute Kalman gain (vector)
			sig_times_a = torch.matmul(S_hh, a_t)  # part of eqn 4 and 5
			
			sigma_t = sigma[t]
			S_vv = torch.matmul(a_t.t(), sig_times_a) + sigma_t.pow(2)  # fourth eqn
			kalman_gain = torch.div(sig_times_a, S_vv+1e-8)  # fifth eqn
			#print(kalman_gain.shape)
			
			# Prediction Error (delta)
			delta = z[t] - b[t] - mu_v  # part of eqn 6
			#print(delta.shape)
			
			# Filtered estimates
			mu_t = mu_h + torch.matmul(kalman_gain, delta)  # sixth eqn
			
			# Joseph's symmetrized update for covariance
			ImKa = eye_h - torch.matmul(kalman_gain, a_t.t())  # identity minus K*a
			S_t = torch.matmul(torch.matmul(ImKa, S_hh), ImKa.t()) \
					+ torch.mul(torch.matmul(kalman_gain, kalman_gain.t()), sigma_t.pow(2)) # seventh eqn
			
			# log likelihood
			log_p = (-0.5 * (delta*delta / (S_vv + 1e-8)
							 + np.log(2.0*np.pi)
							 + torch.log(S_vv + 1e-8))
					)
			
			mu_seq.append(mu_t)
			S_seq.append(S_t)
			log_p_seq.append(log_p)
			deltas.append(delta)
			total_loss += delta**2
			
		return mu_seq, S_seq, log_p_seq, deltas, total_loss

	def reconstruct(self, mu_seq, S_seq):
		a_np = self.a.numpy()
		T = len(mu_seq)
		sigma_np = self.sigma.detach().numpy()
		
		v_filtered_mean = np.array([a_np[:, t].dot(mu_t.detach().numpy()) for t, mu_t in enumerate(mu_seq)]).reshape(T,)
		v_filtered_std = np.sqrt(np.array([a_np[:, t].dot(S_t.detach().numpy()).dot(a_np[:,t]) + np.square(sigma_np[t]) 
										   for t, S_t in enumerate(S_seq)]).reshape((T,)))
	
		return v_filtered_mean, v_filtered_std

	def forecast(self, mu_last_state, S_last_state, F, a, g, sigma, horizon):

		forecasts_mean = []
		forecasts_std = []

		mu_last_state = mu_last_state.detach().numpy()
		S_last_state = S_last_state.detach().numpy()
		F = F.numpy()
		a = a.numpy()
		g = g.detach().numpy()
		sigma = sigma.detach().numpy()

		for t in range(horizon):
			a_t = a[:, t]
			forecast_mean = a_t.dot(mu_last_state)[0]
			forecast_std = a_t.dot(S_last_state).dot(a_t) + np.square(sigma[t])[0]

			forecasts_mean.append(forecast_mean)
			forecasts_std.append(forecast_std)

			mu_last_state = F[:, :, t].dot(mu_last_state)
			S_last_state = F[:, :, t].dot(S_last_state).dot(F[:, :, t].T)

		return np.array(forecasts_mean), np.array(forecasts_std)


	def plot_reconstruction_forecasts(self, v_filtered_mean, v_filtered_std, forecasts_mean, forecasts_std):

		plt.plot(self.obs, color="r")
		plt.plot(v_filtered_mean, color="b")
		T = len(v_filtered_mean)
		x = np.arange(T)
		plt.fill_between(x, v_filtered_mean-v_filtered_std,
						 v_filtered_mean+v_filtered_std,
						 facecolor="blue", alpha=0.2)

		plt.plot(np.arange(T, T+len(forecasts_mean)), forecasts_mean, color="g")
		plt.fill_between(np.arange(T, T+len(forecasts_mean)), forecasts_mean-forecasts_std,
						 forecasts_mean+forecasts_std,
						 facecolor="green", alpha=0.2)

		plt.legend(["data", "reconstruction", "forecasts"]);
		plt.show()


	def forward(self, horizon=12):
		mu_seq, S_seq, nlls, deltas, total_loss = self.ISSM_filter(self.z, self.b, self.F, self.a, self.g, self.sigma, self.m_prior, self.S_prior)
		#print(deltas)
		#reconst_mean, reconst_std = self.reconstruct(mu_seq, S_seq)
		#forecasts_mean, forecasts_std = self.forecast(mu_seq[-1],
										  #S_seq[-1],
										  #self.F, self.a, self.g, self.sigma, horizon=horizon)
		#self.plot_reconstruction_forecasts(reconst_mean, reconst_std, forecasts_mean, forecasts_std)
		#plt.plot(deltas)
		#plt.show()
		return total_loss




		