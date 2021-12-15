import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, 
                 shared_model, 
                 local_model, 
                 optimizer, 
                 global_episode_ctr, 
                 max_episodes,
                 result_queue, 
                 render, 
                 environment, 
                 update_period = 32, 
                ):
        
        super(Worker, self).__init__()
        self.max_episodes = max_episodes
        self.global_episode_ctr = global_episode_ctr
        self.result_queue = result_queue
        self.shared_model = shared_model
        self.local_model = local_model
        self.optimizer = optimizer
        self.env = environment
        self.render = render
        self.update_period = update_period

        
    def update_shared_model(self, state, done, rewards, values, log_probs, entropies):
        # Get value for the next state
        if done: 
            value = torch.zeros(1, 1)
        else:
            logit, value = self.local_model(state.unsqueeze(0))
            value = value.detach()
        values.append(value)

        # Loss calculation
        policy_loss = 0
        value_loss = 0
        R = value
        value_loss_coef = 0.5
        advantage_coef = 0.5
        entropy_coef = 0.005
        gamma = 0.95
        gae = torch.zeros(1, 1)
        gae_lambda = 1

        for i in reversed(range(len(rewards))):
            # Value loss
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation + policy loss
            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma * gae_lambda + delta_t
            policy_loss = policy_loss - log_probs[i] * gae.detach() - entropy_coef * entropies[i]
            
        self.optimizer.zero_grad()
        loss = policy_loss + value_loss_coef * value_loss
        loss.backward()
        for local_params, shared_params in zip(self.local_model.parameters(), self.shared_model.parameters()):
            shared_params._grad = local_params.grad  
            
        self.optimizer.step()

        
    def update_local_model(self):
        # Copy params from shared model to local model. This also zeroes the local model gradients. 
        self.local_model.load_state_dict(self.shared_model.state_dict())

        
    def run(self):
        while self.global_episode_ctr.value < self.max_episodes:
            # New episode
            # Reset variables 
            values = []
            log_probs = []
            rewards = []
            entropies = []
            iteration_count = 0
            episode_rewards = 0 
            
            # Reset environment and get initial state
            state = self.env.reset()
            state = torch.from_numpy(state)
            
            # Get number for current episode and increment the counter
            with self.global_episode_ctr.get_lock():
                current_episode = self.global_episode_ctr.value
                self.global_episode_ctr.value += 1
                
            done = False
            while not done:
                iteration_count += 1
                if self.render:
                    self.env.render()
                
                # Model forward pass
                logit, value = self.local_model(state.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)
                
                # Update environment
                state, reward, done, _ = self.env.step(action.numpy()[0, 0])
                state = torch.from_numpy(state)
                
                # Store values for model backpropagation
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                episode_rewards += reward
                
                # Periodic model backpropagate and update
                if (iteration_count % self.update_period == 0) and (not done): 
                    self.update_shared_model(state, done, rewards, values, log_probs, entropies)
                    self.update_local_model()
                    values = []
                    log_probs = []
                    rewards = []
                    entropies = []
                
            # Episode done 
            results = dict(
                episode=current_episode,
                reward=episode_rewards, 
            )
            self.result_queue.put(results)
            self.update_shared_model(state, done, rewards, values, log_probs, entropies)
            self.update_local_model()
            
        # End of trainign work
        self.result_queue.put(None)