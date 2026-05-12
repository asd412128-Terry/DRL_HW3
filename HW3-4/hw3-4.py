import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy

from Gridworld import Gridworld

# ==========================================
# 1. Noisy Networks for Exploration
# ==========================================
class NoisyLinear(nn.Module):
    # 【修改點 1】: std_init 預設從 0.5 降為 0.1，避免小地圖中探索過度導致無法收斂
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ==========================================
# 2. Distributional Dueling Network
# ==========================================
class RainbowNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_atoms, v_min, v_max):
        super(RainbowNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        # Shared feature layer — 【加速】128 neurons 對 5x5 地圖夠用，比 150 快約 25%
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Advantage stream with NoisyLinear
        self.adv_hidden = NoisyLinear(128, 128, std_init=0.1)
        self.adv_out = NoisyLinear(128, out_dim * num_atoms, std_init=0.1)
        
        # Value stream with NoisyLinear
        self.val_hidden = NoisyLinear(128, 128, std_init=0.1)
        self.val_out = NoisyLinear(128, num_atoms, std_init=0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        feat = self.feature(x)
        
        adv = F.relu(self.adv_hidden(feat))
        val = F.relu(self.val_hidden(feat))
        
        adv = self.adv_out(adv).view(batch_size, self.out_dim, self.num_atoms)
        val = self.val_out(val).view(batch_size, 1, self.num_atoms)
        
        # Combine Dueling Network streams
        q_dist = val + adv - adv.mean(dim=1, keepdim=True)
        prob = F.softmax(q_dist, dim=-1)
        return prob
        
    def reset_noise(self):
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
        self.val_hidden.reset_noise()
        self.val_out.reset_noise()

    def get_action(self, x):
        with torch.no_grad():
            prob = self.forward(x)
            q_values = (prob * self.support).sum(2)
            action = q_values.argmax(1).item()
        return action

# ==========================================
# 3. Prioritized Experience Replay (PER)
# ==========================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def append(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-5

# ==========================================
# 4. Multi-step Learning (N-step returns)
# ==========================================
class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
        
    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        ret = []
        if len(self.buffer) == self.n_step:
            state_n, action_n, _, _, _ = self.buffer[0]
            reward_n = sum([self.gamma**i * self.buffer[i][2] for i in range(self.n_step)])
            _, _, _, next_state_n, done_n = self.buffer[-1]
            ret.append((state_n, action_n, reward_n, next_state_n, done_n))
            
        if done:
            if len(self.buffer) == self.n_step:
                self.buffer.popleft()
            while len(self.buffer) > 0:
                state_n, action_n, _, _, _ = self.buffer[0]
                reward_n = sum([self.gamma**i * self.buffer[i][2] for i in range(len(self.buffer))])
                _, _, _, next_state_n, done_n = self.buffer[-1]
                ret.append((state_n, action_n, reward_n, next_state_n, done_n))
                self.buffer.popleft()
        return ret

# ==========================================
# 5. Core Rainbow Computation
# ==========================================
def compute_loss(model, target_model, batch, indices, weights, gamma_n, support, v_min, v_max, num_atoms):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.stack(states)
    actions = torch.tensor(actions).unsqueeze(1).unsqueeze(2).expand(-1, -1, num_atoms)
    rewards = torch.tensor(rewards).unsqueeze(1).float()
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones).unsqueeze(1).float()
    weights = torch.tensor(weights).unsqueeze(1).float()
    
    with torch.no_grad():
        next_probs_online = model(next_states)
        next_q_online = (next_probs_online * support).sum(2)
        best_actions = next_q_online.argmax(1)
        
        next_probs = target_model(next_states)
        best_actions = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, num_atoms)
        next_probs = next_probs.gather(1, best_actions).squeeze(1)
        
        Tz = rewards + (1 - dones) * gamma_n * support.unsqueeze(0)
        Tz = Tz.clamp(min=v_min, max=v_max)
        
        b = (Tz - v_min) / delta_z
        b = b.clamp(min=0, max=num_atoms - 1) 
        
        l = b.floor().long()
        u = b.ceil().long()
        
        m = torch.zeros(states.size(0), num_atoms)
        offset = torch.linspace(0, (states.size(0) - 1) * num_atoms, states.size(0)).long().unsqueeze(1).expand(states.size(0), num_atoms)
        
        d_m_l = (u.float() - b) * next_probs
        d_m_u = (b - l.float()) * next_probs
        
        eq_mask = (l == u)
        d_m_l[eq_mask] = next_probs[eq_mask]
        d_m_u[eq_mask] = 0.0
        
        m.view(-1).index_add_(0, (l + offset).view(-1), d_m_l.view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), d_m_u.view(-1))
        
    probs = model(states)
    action_probs = probs.gather(1, actions).squeeze(1)
    
    loss = -(m * action_probs.clamp(min=1e-8).log()).sum(1)
    weighted_loss = (weights.squeeze() * loss).mean()
    
    return weighted_loss, loss.detach().numpy()

# ==========================================
# 6. Training Function
# ==========================================
def train_rainbow_dqn(epochs=20000): # 【修改】增加 Epochs 讓模型有時間收斂
    print("--- Training FULL Rainbow DQN for Random Mode GridWorld ---")
    
    # Hyperparameters
    torch.set_num_threads(torch.get_num_threads())  # 【加速】讓 PyTorch 自動使用所有 CPU 核心
    print(f"Using {torch.get_num_threads()} CPU threads")
    
    in_dim = 100
    out_dim = 4
    num_atoms = 51
    # 【修改點 2】擴大 v_min，因為最多走 50 步會扣 50 分，限制在 -20 會讓模型分不出多慘
    v_min, v_max = -50.0, 50.0 
    gamma = 0.99
    n_step = 3
    gamma_n = gamma ** n_step
    batch_size = 64       # 【加速】從 128 降到 64，每次訓練快一倍
    lr = 1e-3
    target_update_freq = 1000  # 【加速】減少 target network 複製次數
    train_every = 4            # 【加速】每 4 步才訓練一次，減少 75% 訓練次數但效果幾乎不變
    
    model = RainbowNet(in_dim, out_dim, num_atoms, v_min, v_max)
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 【修改點 3】加入 Learning Rate Scheduler，讓後期收斂更穩定
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)  # 【修改】放慢衰減，避免後期停止學習
    replay_buffer = PrioritizedReplayBuffer(capacity=50000)  # 【修改】擴大 buffer，讓經驗更多樣
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    losses = []
    win_rates = []
    wins = 0
    step_count = 0
    
    beta = 0.4
    beta_increment = (1.0 - beta) / (epochs * 10)  # 【修改】配合新 epochs 調整 beta 增量
    
    for i in range(epochs):
        game = Gridworld(size=5, mode='random')
        state_np = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        state = torch.from_numpy(state_np).float().squeeze(0)
        
        n_step_buffer = NStepBuffer(n_step, gamma)
        status = 1
        mov = 0
        
        model.reset_noise()
        
        while status == 1:
            step_count += 1
            mov += 1
            
            action_idx = model.get_action(state.unsqueeze(0))
            action = action_set[action_idx]
            
            game.makeMove(action)
            next_state_np = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
            next_state = torch.from_numpy(next_state_np).float().squeeze(0)
            
            reward = game.reward()
            done = True if reward > 0 or reward < -1 else False
            
            # N-step processing
            n_step_exps = n_step_buffer.append(state, action_idx, reward, next_state, done)
            for exp in n_step_exps:
                s_n, a_n, r_n, ns_n, d_n = exp
                replay_buffer.append(s_n, a_n, r_n, ns_n, d_n)
                
            state = next_state
            
            if len(replay_buffer.buffer) >= batch_size and step_count % train_every == 0:  # 【加速】每 4 步訓練一次
                beta = min(1.0, beta + beta_increment)
                batch, indices, weights = replay_buffer.sample(batch_size, beta)
                
                loss, td_errors = compute_loss(
                    model, target_model, batch, indices, weights, 
                    gamma_n, model.support, v_min, v_max, num_atoms
                )
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
                replay_buffer.update_priorities(indices, td_errors)
                losses.append(loss.item())
                
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
            if reward != -1 or mov > 50:
                status = 0
                if reward > 0:
                    wins += 1
                    
        scheduler.step()
                    
        if (i+1) % 100 == 0:
            win_rates.append(wins / 100.0)
            wins = 0
            print(f"Rainbow Epoch {i+1}/{epochs} | Win Rate (last 100): {win_rates[-1]:.2f}")
            
    return model, losses, win_rates

def test_rainbow_model(model, game, display=False):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    state_np = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
    path = [game.board.components['Player'].pos]
    status, i = 1, 0
    model.eval() # 測試時關閉 NoisyNet 的隨機性，使用平均權重
    
    while status == 1:
        state = torch.from_numpy(state_np).float()
        action_idx = model.get_action(state)
        action = action_set[action_idx]
        
        if display: print(f'Move #: {i}; Taking action: {action}')
        game.makeMove(action)
        path.append(game.board.components['Player'].pos)
        state_np = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        
        reward = game.reward()
        if reward != -1:
            status = 2 if reward > 0 else 0
        i += 1
        # 【修改點 4】放寬測試步數限制到 50 步，避免它只是在閃牆壁就被迫停止
        if i > 50: break 
            
    return (status == 2), path

def plot_rainbow_grid_and_path(path, initial_components, win):
    fig, ax = plt.subplots(figsize=(6, 6))
    goal = initial_components['Goal']
    pit = initial_components['Pit']
    wall = initial_components['Wall']
    start = initial_components['Player']
    
    ax.set_xticks(np.arange(-0.5, 5, 1))
    ax.set_yticks(np.arange(-0.5, 5, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    for r in range(5):
        for c in range(5):
            rect, text = None, None
            if (r, c) == goal:
                rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='lightgreen')
                text = 'Goal'
            elif (r, c) == pit:
                rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='salmon')
                text = 'Pit'
            elif (r, c) == wall:
                rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='gray')
                text = 'Wall'
            elif (r, c) == start:
                rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='lightyellow')
                text = 'Start'
            
            if rect:
                ax.add_patch(rect)
                ax.text(c, 4-r, text, va='center', ha='center', fontsize=12, fontweight='bold')
    
    if path:
        x = [pos[1] for pos in path]
        y = [4 - pos[0] for pos in path]
        ax.plot(x, y, marker='o', color='purple', linewidth=3, markersize=8, label='Rainbow DQN Path')
        
    outcome = '✔ Win' if win else '✘ Lose'
    ax.set_title(f'Rainbow DQN Path [{outcome}]', fontsize=14)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.legend(loc='upper right')
    
    plt.savefig('hw3-4_rainbow_path.jpg', dpi=300, bbox_inches='tight')
    print("Path plot saved as hw3-4_rainbow_path.jpg")

if __name__ == '__main__':
    # Train Rainbow
    epochs = 20000  # 【修改】從 3000 增加到 20000
    model, losses, win_rates = train_rainbow_dqn(epochs=epochs)
    
    # Plot Metrics
    def moving_average(a, n=100):
        if len(a) < n: return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(losses) > 100:
        plt.plot(moving_average(losses, 500), label='Rainbow DQN Loss', color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Moving Avg)')
    plt.title('Loss over Training Steps')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    x_epochs = np.arange(100, len(win_rates)*100 + 1, 100)
    plt.plot(x_epochs, win_rates, marker='^', label='Rainbow DQN', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Win Rate (per 100 epochs)')
    plt.title('Win Rate over Epochs (Random Mode)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw3-4_rainbow_metrics.png', dpi=300)
    print("Metrics plot saved as hw3-4_rainbow_metrics.png")
    
    # 測試環節：自動尋找一個能贏的隨機地圖來畫圖
    print("\nTesting trained Rainbow model (Looking for a winning path for the plot)...")
    success = False
    attempts = 0
    while not success and attempts < 100:
        attempts += 1
        game = Gridworld(size=5, mode='random')
        initial_components = {
            'Player': game.board.components['Player'].pos,
            'Goal': game.board.components['Goal'].pos,
            'Pit': game.board.components['Pit'].pos,
            'Wall': game.board.components['Wall'].pos
        }
        win, path = test_rainbow_model(model, game, display=False)
        if win:
            success = True
            
    print(f"Rainbow DQN won: {win} (after {attempts} attempt(s))")
    plot_rainbow_grid_and_path(path, initial_components, win)