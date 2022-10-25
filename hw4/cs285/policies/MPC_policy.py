import numpy as np
import scipy.stats as stats

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
            #for i in range(num_sequences * horizon):
            #    sequence_num = i // horizon
            #    horizon_idx = i % horizon
            #    random_action_sequences[sequence_num, horizon_idx, :] = self.ac_space.sample()
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            mu = 0.5 * (self.high - self.low)
            mu = np.repeat(mu.reshape(1, -1), horizon, axis=0).reshape(1, -1) # [1, h*ac_dim]
            std = ((self.high - self.low) ** 2) / 12.
            std = np.repeat(std.reshape(1, -1), horizon, axis=0).reshape(1, -1)
            #lows = (np.repeat(self.low.reshape(1, -1), horizon, axis=0).reshape(1, -1) - mu) / std
            #highs = (np.repeat(self.high.reshape(1, -1), horizon, axis=0).reshape(1, -1) - mu) / std
            for i in range(self.cem_iterations):
                if i == 0:
                    action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
                else:
                    action_sequences = stats.truncnorm.rvs(lows, highs, loc=mu, scale=std, size=(num_sequences, horizon * self.ac_dim)).reshape((num_sequences, horizon, self.ac_dim))
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                rewards = -self.evaluate_candidate_sequences(action_sequences, obs)
                indices = np.argpartition(rewards, self.cem_num_elites - 1) # top elite rewards, since argsort sorts by minimal
                elite_sequences = action_sequences[indices][:self.cem_num_elites].reshape((-1, horizon * self.ac_dim))
                next_mu = np.mean(elite_sequences, axis=0)
                next_std = np.std(elite_sequences, axis=0)
                mu = self.cem_alpha * next_mu + (1. - self.cem_alpha) * mu
                std = self.cem_alpha * next_std + (1. - self.cem_alpha) * std
                lows = (np.repeat(self.low.reshape(1, -1), horizon, axis=0).reshape(1, -1) - mu) / std
                highs = (np.repeat(self.high.reshape(1, -1), horizon, axis=0).reshape(1, -1) - mu) / std
                #action_sequences = 

            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            cem_action = mu.reshape((horizon, self.ac_dim))

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        ensembles_num = 0
        total_rewards = 0.
        for model in self.dyn_models: 
            ensembles_num += 1
            total_rewards += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
        return total_rewards / ensembles_num

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = np.argmax(predicted_rewards)
            action_to_take = candidate_action_sequences[best_action_sequence, 0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        sum_of_rewards = 0.  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        N, H, D_action = candidate_action_sequences.shape
        obs = np.repeat(obs.reshape((1, -1)), N, axis=0)
        for h in range(H):
            predicted_obs = model.get_prediction(obs, candidate_action_sequences[:, h, :], self.data_statistics)
            rewards, dones = self.env.get_reward(predicted_obs, candidate_action_sequences[:, h, :])
            sum_of_rewards += rewards
            obs = predicted_obs
        return sum_of_rewards
