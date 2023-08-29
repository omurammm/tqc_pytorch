import torch

from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE
from scipy.stats import norm
import numpy as np
import wandb


class Trainer(object):
	def __init__(
		self,
		args,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		bottom_quantiles_to_drop,
		move_mean_quantiles,
		move_mean_from_origin,
		target_entropy,
		ens_type,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.discount = discount
		self.tau = tau

		self.ens_type = ens_type
		
		# if ens_type in ['ave', 'sample']:
		if ens_type != 'tqc':
			self.top_quantiles_to_drop = top_quantiles_to_drop
			self.quantiles_total = critic.n_quantiles
		else:
			self.top_quantiles_to_drop = top_quantiles_to_drop * critic.n_nets
			self.quantiles_total = critic.n_quantiles * critic.n_nets
		# self.bottom_quantiles_to_drop = bottom_quantiles_to_drop
		# self.move_mean_quantiles = move_mean_quantiles
		# self.move_mean_from_origin = move_mean_from_origin

		self.target_entropy = target_entropy

		# self.qem = args.qem
		if ens_type in ['qem_wls', 'qem_ols']:
			quantile_tau = np.arange(critic.n_quantiles) / critic.n_quantiles + 1 / 2 / critic.n_quantiles
			tau_expand = np.array([t for _ in range(critic.n_nets) for t in quantile_tau])
			if args.reg_dim == 4:
				X = [[1, norm.ppf(t), norm.ppf(t)**2 - 1, norm.ppf(t)**3 - 3 * norm.ppf(t)] for t in tau_expand]
				self.X_pred = [[1, norm.ppf(t), norm.ppf(t)**2 - 1, norm.ppf(t)**3 - 3 * norm.ppf(t)] for t in quantile_tau]
			elif args.reg_dim == 3:
				X = [[1, norm.ppf(t), norm.ppf(t)**2 - 1] for t in tau_expand]
				self.X_pred = [[1, norm.ppf(t), norm.ppf(t)**2 - 1] for t in quantile_tau]
			elif args.reg_dim == 2:
				X = [[1, norm.ppf(t)] for t in tau_expand]
				self.X_pred = [[1, norm.ppf(t)] for t in quantile_tau]
			else:
				raise ValueError
			self.X = torch.tensor(X, device=DEVICE).float()
			self.n_quantiles = critic.n_quantiles
			self.n_nets = critic.n_nets
			self.no_sort_qem = args.no_sort_qem
		
		

		self.total_it = 0

	def train(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			if self.ens_type == 'ave':
				sorted_z, _ = torch.sort(next_z, dim=2) # (batch, nets, tiles)
				sorted_z = torch.mean(sorted_z, dim=1) # (batch, tiles)
			elif self.ens_type == 'sample':
				# TODO: refactor
				sorted_z, _ = torch.sort(next_z, dim=2) # (batch, nets, tiles)
				zero_z = torch.zeros((sorted_z.shape[0], sorted_z.shape[2]), dtype=sorted_z.dtype).to(sorted_z.device) # (batch, tiles)
				for b in range(sorted_z.shape[0]):
					for t in range(sorted_z.shape[2]):
						idx = torch.randint(0, sorted_z.shape[1], (1, ))[0]
						zero_z[b, t] = sorted_z[b, idx, t]
				sorted_z = zero_z
			elif self.ens_type == 'qem_wls':
				if self.no_sort_qem:
					sorted_z = next_z
				else:
					sorted_z, _ = torch.sort(next_z, dim=2) # (batch, nets, tiles)
				var = sorted_z.var(1) # (batch, tiles,)
				var = var.repeat(1, self.n_nets) # (batch, tiles*n_nets,)
				weight = 1 / (var + 1e-8)
				Q = sorted_z.reshape(batch_size, -1).unsqueeze(-1) # (batch, tiles*n_nets, 1)
				X = self.X.expand(batch_size, -1, -1) # (batch, tiles*n_nets, qem_dim)
				W = torch.diag_embed(weight) # (batch, tiles*n_nets, tiles*n_nets)
				M = (X.transpose(1, 2).bmm(W).bmm(X)).inverse().bmm(X.transpose(1, 2)).bmm(W).bmm(Q) # (batch, qem_dim, 1)
				X_pred = self.X.expand(batch_size, -1, -1) # (batch, tiles, qem_dim)
				sorted_z = X_pred.bmm(M) # (batch, tiles, 1)
				sorted_z = sorted_z.squeeze(-1) # (batch, tiles)

			elif self.ens_type == 'qem_ols':
				if self.no_sort_qem:
					sorted_z = next_z
				else:
					sorted_z, _ = torch.sort(next_z, dim=2) # (batch, nets, tiles)
				Q = sorted_z.reshape(batch_size, -1).unsqueeze(-1) # (batch, tiles*n_nets, 1)
				X = self.X.expand(batch_size, -1, -1) # (batch, tiles*n_nets, qem_dim)
				M = (X.transpose(1, 2).bmm(X)).inverse().bmm(X.transpose(1, 2)).bmm(Q) # (batch, qem_dim, 1)
				X_pred = self.X.expand(batch_size, -1, -1) # (batch, tiles, qem_dim)
				sorted_z = X_pred.bmm(M) # (batch, tiles, 1)
				sorted_z = sorted_z.squeeze(-1) # (batch, tiles)


			elif self.ens_type == 'tqc':
				sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			else:
				raise ValueError
			# sorted_z_part = sorted_z[:, self.bottom_quantiles_to_drop:self.quantiles_total-self.top_quantiles_to_drop]
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
			# center_quantile_idx = (sorted_z_part.shape[1] // 2) + 1
			# center_quantile = sorted_z_part[:, center_quantile_idx]
			# move_target_quantile = sorted_z_part[:, center_quantile_idx - self.move_mean_quantiles]
			# move_diff = center_quantile - move_target_quantile
			# move_diff = move_diff.unsqueeze(1).repeat(1, sorted_z_part.shape[1])

			# if self.move_mean_from_origin:
			# 	mean_diff = torch.mean(sorted_z_part, dim=1) - torch.mean(sorted_z, dim=1)
			# 	mean_diff = mean_diff.unsqueeze(1).repeat(1, sorted_z_part.shape[1])
			# 	sorted_z_part = sorted_z_part - mean_diff
			# print(sorted_z_part)
			# print(sorted_z_part.shape)
			# print('center')
			# print(center_quantiles, center_quantiles.shape)
			# print('target')
			# print(move_target_quantiles, move_target_quantiles.shape)
			# print('diff')
			# print(move_diff, move_diff.shape)

			# sorted_z_part = sorted_z_part - move_diff

			# mean = torch.mean(sorted_z_part, dim=1).unsqueeze(1).repeat(1, sorted_z_part.shape[1])
			# sorted_z_part = sorted_z_part - (mean * self.move_mean_quantiles)

			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

		if self.ens_type == 'qem_wls':
			z = self.critic(state, new_action) # (batch, net, tiles)
			sorted_z, _ = torch.sort(z, dim=2) # (batch, nets, tiles)
			var = sorted_z.var(1) # (batch, tiles,)
			var = var.repeat(1, self.n_nets) # (batch, tiles*n_nets,)
			weight = 1 / (var + 1e-8)
			Q = sorted_z.reshape(batch_size, -1).unsqueeze(-1) # (batch, tiles*n_nets, 1)
			X = self.X.expand(batch_size, -1, -1) # (batch, tiles*n_nets, qem_dim)
			W = torch.diag_embed(weight) # (batch, tiles*n_nets, tiles*n_nets)
			M = (X.transpose(1, 2).bmm(W).bmm(X)).inverse().bmm(X.transpose(1, 2)).bmm(W).bmm(Q) # (batch, qem_dim, 1)
			q_qem = M[:, 0] # (batch, 1)
			actor_loss = (alpha * log_pi - q_qem).mean()
			if self.total_it % 1000 == 0:
				log_dict = {}
				for i in range(self.n_quantiles):
					log_dict[f'var_for_tiles_{str(i).zfill(2)}'] = float(var[0][i])
				wandb.log(log_dict, step=self.total_it)

		elif self.ens_type == 'qem_ols':
			z = self.critic(state, new_action) # (batch, net, tiles)
			sorted_z, _ = torch.sort(z, dim=2) # (batch, nets, tiles)
			Q = sorted_z.reshape(batch_size, -1).unsqueeze(-1) # (batch, tiles*n_nets, 1)
			X = self.X.expand(batch_size, -1, -1) # (batch, tiles*n_nets, qem_dim)
			M = (X.transpose(1, 2).bmm(X)).inverse().bmm(X.transpose(1, 2)).bmm(Q) # (batch, qem_dim, 1)
			q_qem = M[:, 0] # (batch, 1)
			actor_loss = (alpha * log_pi - q_qem).mean()
		else:
			actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.total_it += 1

	def save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=DEVICE))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target", map_location=DEVICE))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=DEVICE))
		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=DEVICE))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=DEVICE))
		self.log_alpha = torch.load(filename + '_log_alpha', map_location=DEVICE)
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer", map_location=DEVICE))
