from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: get this from previous HW
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW
        # if not sampling return the mean of the distribution 
        #print(self.action_range)
        observation = ptu.from_numpy(obs)
        action_distribution = self.forward(observation)
        if sample:
            action = action_distribution.rsample(sample_shape=(self.ac_dim, ))
        else:
            action = action_distribution.mean
        high, low = self.action_range[1], self.action_range[0]
        range = high - low
        action = 0.5 * (1. + action) * range + low
        #print(high, low, range)
        #print('transformed_mu', (1. + action_distribution.mean) * range / 2. + low)
        #print((1. + action_distribution.mean) * range / 2. - low)
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        mu = self.mean_net.forward(observation)
        #print('mu:', mu)
        std_min, std_max = self.log_std_bounds[0], self.log_std_bounds[1]
        stds = torch.exp(torch.clip(self.logstd, min=std_min, max=std_max))
        action_distribution = sac_utils.SquashedNormal(loc=mu, scale=stds)
        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        return action_distribution

    def update(self, obs, critic):
        # TODO: get this from previous HW
        action_distribution = self.forward(obs)
        actions = action_distribution.rsample()
        high, low = self.action_range[1], self.action_range[0]
        range = high - low
        resized_actions = (1. + actions) * range / 2. + low
        q_1, q_2 = critic(obs, resized_actions)
        q_min = torch.min(q_1, q_2).squeeze()
        log_probs = action_distribution.log_prob(actions).sum(dim=1)

        actor_loss = (self.alpha.detach() * log_probs - q_min).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha