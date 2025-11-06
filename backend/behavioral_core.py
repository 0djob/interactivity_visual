import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import random
import copy
import numpy as np

GLOBAL_COMPLEXITY_FACTOR = torch.randn(1) * 0.1

class BehavioralSelfPredictor(nn.Module):
    
    def __init__(self, 
                 behavior_size=64,
                 hidden_size=64, 
                 horizon=20):
        super().__init__()
        
        self.observation_size = behavior_size
        self.action_size = 4
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.behavior_size = behavior_size
        
        self.lstm = nn.LSTM(
            input_size=self.observation_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.behavior_size)
        )
        
        self.target_lstm = copy.deepcopy(self.lstm)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_value = copy.deepcopy(self.value)
        
        for param in self.target_lstm.parameters():
            param.requires_grad = False
        for param in self.target_policy.parameters():
            param.requires_grad = False
        for param in self.target_value.parameters():
            param.requires_grad = False
        
        self._initialize_weights()
        
        online_params = (list(self.lstm.parameters()) + 
                        list(self.policy.parameters()) + 
                        list(self.value.parameters()))
        
        self.optimizer = optim.Adam(online_params, lr=0.002) 
        
        self.hidden = self._init_hidden()
        self.target_hidden = self._init_hidden()
        self.current_behavior_state = torch.randn(self.behavior_size) * 0.8
        
        self.observation_buffer = []
        self.target_buffer = []
        
        self.behavior_history = []
        self.max_history = 100
        
        self.step_count = 0
        self.i_score_history = []
        self.conditional_complexity_history = []
        self.semiconditional_complexity_history = []
        self.reward_history = []
        self.curiosity_history = []
        self.novelty_history = []
        
        self.complexity_scale = 20.0
        self.update_count = 0
        self.last_i_score = 0.0
        
        self.curiosity_weight = 0.5       
        self.novelty_weight = 0.3         
        self.entropy_weight = 0.1         
        self.min_i_score = 0.015          
        self.negative_penalty = 5.0       
        
        self.consecutive_low_i_score = 0
        self.reset_threshold = 50         
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
    
    def _sync_target(self, tau=0.01):
        """
        Gradually updates target network to match online network.
        tau: How much to update (0.01 = 1% update)
        """
        for target_param, online_param in zip(
            self.target_lstm.parameters(), 
            self.lstm.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )

        for target_param, online_param in zip(
            self.target_policy.parameters(),
            self.policy.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )

        for target_param, online_param in zip(
            self.target_value.parameters(),
            self.value.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )

        self.target_hidden = (
            tau * self.hidden[0] + (1 - tau) * self.target_hidden[0],
            tau * self.hidden[1] + (1 - tau) * self.target_hidden[1]
        )
    
    def forward_online(self, observation, hidden=None):
        """Forward through online network."""
        if hidden is None:
            hidden = self.hidden
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0).unsqueeze(0)
        elif observation.dim() == 2:
            observation = observation.unsqueeze(1)
        
        lstm_out, new_hidden = self.lstm(observation, hidden)
        lstm_out_squeezed = lstm_out.squeeze(1)
        
        action_logits = self.policy(lstm_out_squeezed)
        state_value = self.value(lstm_out_squeezed)
        
        return action_logits.squeeze(0), state_value.squeeze(0), new_hidden, lstm_out_squeezed.squeeze(0)
    
    def forward_target(self, observation, hidden=None):
        """Process observation through target network to get predictions."""
        if hidden is None:
            hidden = self.target_hidden
        
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0).unsqueeze(0)
            elif observation.dim() == 2:
                observation = observation.unsqueeze(1)
            
            lstm_out, new_hidden = self.target_lstm(observation, hidden)
            lstm_out_squeezed = lstm_out.squeeze(1)
            
            action_logits = self.target_policy(lstm_out_squeezed)
            state_value = self.target_value(lstm_out_squeezed)
            
            return action_logits.squeeze(0), state_value.squeeze(0), new_hidden, lstm_out_squeezed.squeeze(0)

    def step_environment(self, action):
        """Simulate environment response to action."""
        global GLOBAL_COMPLEXITY_FACTOR
        
        if action == 0: GLOBAL_COMPLEXITY_FACTOR += 2.0
        elif action == 1: GLOBAL_COMPLEXITY_FACTOR -= 1.0
        elif action == 2: GLOBAL_COMPLEXITY_FACTOR += 1.5
        elif action == 3: GLOBAL_COMPLEXITY_FACTOR *= 1.2
        
        GLOBAL_COMPLEXITY_FACTOR += torch.randn(1) * 0.5 
        
        reward_noise = (torch.sin(GLOBAL_COMPLEXITY_FACTOR * 10) * 2.0 + 
                       torch.cos(GLOBAL_COMPLEXITY_FACTOR * 5) * 1.5 +
                       torch.randn(1) * (3 + torch.abs(GLOBAL_COMPLEXITY_FACTOR * 2.0)))
        
        done = random.random() < 0.01
        if done: 
            GLOBAL_COMPLEXITY_FACTOR = torch.randn(1) * 1.0
        
        return reward_noise, done

    def _compute_novelty(self, behavior):
        """Calculate how different this behavior is from recent history."""
        if len(self.behavior_history) < 5:
            return 1.0
        
        recent_behaviors = torch.stack(self.behavior_history[-30:]) 
        distances = torch.norm(recent_behaviors - behavior.unsqueeze(0), dim=1)
        min_distance = distances.min().item()
        
        novelty = min_distance / 1.5 
        return min(1.0, novelty)

    def step(self, behavior_input=None):
        """Take one step forward, processing input and generating next action."""
        self.step_count += 1
        
        if behavior_input is None:
            obs = self.current_behavior_state
        else:
            obs = behavior_input
        
        action_logits, V_s, online_hidden_next, new_behavior = self.forward_online(obs, self.hidden)
        
        exploration_noise = 0.3
        
        with torch.no_grad():
            action_probs = torch.softmax(action_logits, dim=-1)
            action_probs = action_probs + exploration_noise
            action_probs = action_probs / action_probs.sum()
            action = torch.multinomial(action_probs, 1).item()
        
        reward, done = self.step_environment(action)
        
        next_obs = new_behavior.detach() + torch.randn_like(new_behavior) * 0.3
        
        self.observation_buffer.append(obs.detach().clone())
        self.target_buffer.append(next_obs.clone())
        
        self.behavior_history.append(next_obs.clone())
        if len(self.behavior_history) > self.max_history:
            self.behavior_history.pop(0)
        
        novelty = self._compute_novelty(next_obs)
        self.novelty_history.append(novelty)
        
        self.hidden = (online_hidden_next[0].detach(), online_hidden_next[1].detach())
        self.current_behavior_state = next_obs
        
        i_score = self.last_i_score 
        learning_loss = 0.0
        curiosity = 0.0
        
        if len(self.observation_buffer) >= self.horizon:
            i_score, learning_loss, curiosity = self._optimize()
            self.last_i_score = i_score
            
            if i_score < 0.005:
                self.consecutive_low_i_score += 1
                self.curiosity_weight = min(1.0, self.curiosity_weight * 1.05)
                self.novelty_weight = min(0.8, self.novelty_weight * 1.05)
            else:
                self.consecutive_low_i_score = max(0, self.consecutive_low_i_score - 1)
            
            if self.consecutive_low_i_score >= self.reset_threshold:
                print(f"\n🔄 Step {self.step_count}: Resetting learning state")
                self._perturbation_reset()
                self.consecutive_low_i_score = 0
            
            if self.update_count % self.horizon == 0 and self.update_count > 0:
                self._sync_target()
        
        self.reward_history.append(reward.item())
        self.curiosity_history.append(curiosity)
        
        if done:
            self.hidden = self._init_hidden()
            self.target_hidden = self._init_hidden()
            self.current_behavior_state = torch.randn(self.behavior_size) * 0.8
        
        return {
            'behavior': self.current_behavior_state,
            'i_score': i_score,
            'conditional_complexity': (self.conditional_complexity_history[-1]
                                      if self.conditional_complexity_history else 0.0),
            'semiconditional_complexity': (self.semiconditional_complexity_history[-1]
                                          if self.semiconditional_complexity_history else 0.0),
            'value_prediction': V_s.detach(),
            'learning_loss': learning_loss,
            'reward': reward.item(),
            'action': action,
            'novelty': novelty,
            'curiosity': curiosity,
            'curiosity_weight': self.curiosity_weight,
            'novelty_weight': self.novelty_weight,
        }
    
    def _perturbation_reset(self):
        """Reset agent state with random noise to encourage exploration."""
        self.current_behavior_state += torch.randn_like(self.current_behavior_state) * 1.0
        
        self.hidden = self._init_hidden()
        self.target_hidden = self._init_hidden()
        
        self.observation_buffer = self.observation_buffer[-10:]
        self.target_buffer = self.target_buffer[-10:]
        
        self.curiosity_weight = 0.8
        self.novelty_weight = 0.5
    
    def _optimize(self):
        """Update network weights to improve predictions."""
        self.update_count += 1
        
        observations = torch.stack(self.observation_buffer[-self.horizon:])
        targets = torch.stack(self.target_buffer[-self.horizon:])
        
        td_cond_list = []
        temp_hidden = (self.hidden[0].detach().clone(), self.hidden[1].detach().clone())
        
        for i in range(self.horizon):
            obs = observations[i]
            target = targets[i]
            
            _, V_s, temp_hidden, _ = self.forward_online(obs, temp_hidden)
            delta_cond = target - V_s
            td_cond_list.append(delta_cond)
        
        td_cond = torch.stack(td_cond_list)
        
        td_semicond_list = []
        temp_target_hidden = (self.target_hidden[0].clone(), self.target_hidden[1].clone())
        
        with torch.no_grad():
            for i in range(self.horizon):
                obs = observations[i]
                target = targets[i]
                
                _, V_s_target, temp_target_hidden, _ = self.forward_target(obs, temp_target_hidden)
                delta_semicond = target - V_s_target
                td_semicond_list.append(delta_semicond)
        
        td_semicond = torch.stack(td_semicond_list)
        
        scale = self.complexity_scale
        
        conditional_complexity = scale * torch.sum(td_cond ** 2)
        semiconditional_complexity = scale * torch.sum(td_semicond ** 2)
        
        i_score_tensor = (semiconditional_complexity - conditional_complexity) / (self.horizon * self.behavior_size)
        
        curiosity_bonus = torch.mean(torch.norm(td_cond, dim=1))
        
        recent_novelty = sum(self.novelty_history[-self.horizon:]) / self.horizon if self.novelty_history else 0.5
        novelty_bonus = torch.tensor(recent_novelty)
        
        action_logits, _, _, _ = self.forward_online(self.current_behavior_state, self.hidden)
        action_probs = torch.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        
        if i_score_tensor.item() < self.min_i_score:
            floor_penalty = (self.min_i_score - i_score_tensor) * 20.0 
        else:
            floor_penalty = torch.tensor(0.0)
        
        if i_score_tensor.item() < 0:
            negative_penalty = -i_score_tensor * self.negative_penalty
        else:
            negative_penalty = torch.tensor(0.0)
        
        loss = (-i_score_tensor +                                  
                -self.curiosity_weight * curiosity_bonus +         
                -self.novelty_weight * novelty_bonus +             
                -self.entropy_weight * entropy +                   
                floor_penalty +                                    
                negative_penalty)                                  
        
        reg_loss = 0.00005 * sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)
        total_loss = loss + reg_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.5)  
        self.optimizer.step()
        
        i_score_val = i_score_tensor.item()
        self.i_score_history.append(i_score_val)
        self.conditional_complexity_history.append(conditional_complexity.item() / (self.horizon * self.behavior_size))
        self.semiconditional_complexity_history.append(semiconditional_complexity.item() / (self.horizon * self.behavior_size))
        
        if len(self.observation_buffer) > self.horizon:
            self.observation_buffer = self.observation_buffer[-self.horizon:]
            self.target_buffer = self.target_buffer[-self.horizon:]
        
        return i_score_val, total_loss.item(), curiosity_bonus.item()
    
    def get_metrics(self) -> Dict:
        """Get current performance measurements."""
        recent_window = min(100, len(self.i_score_history))
        
        i_score_trend = (sum(self.i_score_history[-recent_window:]) / recent_window
                        if recent_window > 0 else 0.0)
        
        avg_novelty = (sum(self.novelty_history[-100:]) / min(100, len(self.novelty_history))
                      if self.novelty_history else 0.0)
        
        avg_curiosity = (sum(self.curiosity_history[-100:]) / min(100, len(self.curiosity_history))
                        if self.curiosity_history else 0.0)
        
        if len(self.i_score_history) >= 20:
            recent_i_scores = self.i_score_history[-20:]
            i_score_variance = np.var(recent_i_scores)
        else:
            i_score_variance = 0.0
        
        return {
            'i_score_history': self.i_score_history[-1000:],
            'conditional_history': self.conditional_complexity_history[-1000:],
            'semiconditional_history': self.semiconditional_complexity_history[-1000:],
            'reward_history': self.reward_history[-1000:],
            'novelty_history': self.novelty_history[-1000:],
            'curiosity_history': self.curiosity_history[-1000:],
            
            'current_i_score': self.i_score_history[-1] if self.i_score_history else 0.0,
            'i_score_trend': i_score_trend,
            'i_score_variance': i_score_variance,
            'avg_novelty': avg_novelty,
            'avg_curiosity': avg_curiosity,
            
            'step_count': self.step_count,
            'global_complexity': GLOBAL_COMPLEXITY_FACTOR.item(),
            
            'in_dark_room': i_score_trend < 0.008 and avg_novelty < 0.15 and self.step_count > 100,
            
            'behavior': self.current_behavior_state.tolist(),
            'behavior_norm': torch.norm(self.current_behavior_state).item(),
            
            'curiosity_weight': self.curiosity_weight,
            'novelty_weight': self.novelty_weight,
            'consecutive_low': self.consecutive_low_i_score,
        }
    
    def reset(self):
        """Reset."""
        global GLOBAL_COMPLEXITY_FACTOR
        GLOBAL_COMPLEXITY_FACTOR = torch.randn(1) * 1.0
        
        self.hidden = self._init_hidden()
        self.target_hidden = self._init_hidden()
        self.current_behavior_state = torch.randn(self.behavior_size) * 0.8
        self.observation_buffer.clear()
        self.target_buffer.clear()
        self.behavior_history.clear()
        self.last_i_score = 0.0
        self.consecutive_low_i_score = 0
        self.curiosity_weight = 0.5
        self.novelty_weight = 0.3


if __name__ == "__main__":
    print("="*75)
    print("AGGRESSIVE ANTI-DARK-ROOM Agent")
    print("="*75)
    
    agent = BehavioralSelfPredictor(
        behavior_size=64,
        hidden_size=64,
        horizon=20
    )
    
    print(f"\n⚡ AGGRESSIVE Settings:")
    print(f"   Curiosity weight: {agent.curiosity_weight} (5x normal)")
    print(f"   Novelty weight: {agent.novelty_weight} (6x normal)")
    print(f"   Entropy weight: {agent.entropy_weight} (5x normal)")
    print(f"   Min I-Score floor: {agent.min_i_score}")
    print(f"   Auto-reset after: {agent.reset_threshold} low steps")
    
    print("\nRunning 500 steps...\n")
    print("Step | I-Score  | Trend    | Novelty | Curiosity | CurW  | Dark?")
    print("-"*80)
    
    behavior = torch.randn(64) * 0.8
    
    for step in range(500):
        result = agent.step(behavior)
        behavior = result['behavior']
        
        if (step + 1) % 50 == 0:
            metrics = agent.get_metrics()
            dark_room = "YES" if metrics['in_dark_room'] else "NO"
            print(f"{step+1:4d} | {result['i_score']:8.5f} | "
                  f"{metrics['i_score_trend']:8.5f} | "
                  f"{result['novelty']:7.4f} | "
                  f"{result['curiosity']:9.4f} | "
                  f"{result['curiosity_weight']:.2f} | "
                  f"{dark_room:4s}")
    
    print("\n" + "="*80)
    metrics = agent.get_metrics()
    print(f"\n📊 Final Results:")
    print(f"   I-Score Trend: {metrics['i_score_trend']:.5f}")
    print(f"   I-Score Variance: {metrics['i_score_variance']:.6f}")
    print(f"   Avg Novelty: {metrics['avg_novelty']:.4f}")
    print(f"   Avg Curiosity: {metrics['avg_curiosity']:.4f}")
    print(f"   Final Curiosity Weight: {metrics['curiosity_weight']:.3f}")
    print(f"   In Dark Room: {metrics['in_dark_room']}")
    
    if not metrics['in_dark_room'] and metrics['i_score_trend'] > 0.010:
        print("\n✅ SUCCESS: Sustained high I-Score!")
    else:
        print("\n⚠️  May need environment complexity (cellular automaton)")