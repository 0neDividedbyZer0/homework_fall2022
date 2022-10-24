from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure.sac_utils import *
import torch
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        with torch.no_grad():
            action_dist = self.actor.forward(next_ob_no)
            a_t1 = action_dist.rsample()
            #print(action_dist)
            #print(action_dist.log_prob(a_t1[0, :]).squeeze(), a_t1.shape)
            high, low = self.action_range[1], self.action_range[0]
            range = high - low
            resized_action = (1. + a_t1) * range / 2. + low
            q_target_1, q_target_2 = self.critic_target.forward(next_ob_no, resized_action)
            min_q = torch.min(q_target_1, q_target_2).squeeze()
            target_val = re_n + self.gamma * (1. - terminal_n) * (min_q - self.actor.alpha * action_dist.log_prob(a_t1).sum(dim=1))

        #print(target_val.shape, min_q.shape, (min_q - self.actor.alpha * action_dist.log_prob(a_t1).squeeze()).shape)
        q_curr_1, q_curr_2 = self.critic.forward(ob_no, ac_na)
        critic_loss = self.critic.loss(q_curr_1.squeeze(), target_val) + self.critic.loss(q_curr_2.squeeze(), target_val)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        ac_na = ptu.from_numpy(ac_na)
        terminal_n = ptu.from_numpy(terminal_n)

        critical_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        soft_update_params(self.critic, self.critic_target, self.critic_tau)
        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)
        self.training_step += 1
        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critical_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
