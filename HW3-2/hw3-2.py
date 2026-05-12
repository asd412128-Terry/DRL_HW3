import numpy as np
import torch
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
from Gridworld import Gridworld

class BasicDQN(torch.nn.Module):
    def __init__(self, l1=100, l2=150, l3=100, l4=4):
        super(BasicDQN, self).__init__()
        self.fc1 = torch.nn.Linear(l1, l2)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(l2, l3)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(l3, l4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQN(torch.nn.Module):
    def __init__(self, l1=100, l2=150, l3=100, l4=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(l1, l2)
        self.relu1 = torch.nn.ReLU()
        
        # Value stream
        self.val_fc1 = torch.nn.Linear(l2, l3)
        self.val_relu1 = torch.nn.ReLU()
        self.val_fc2 = torch.nn.Linear(l3, 1)
        
        # Advantage stream
        self.adv_fc1 = torch.nn.Linear(l2, l3)
        self.adv_relu1 = torch.nn.ReLU()
        self.adv_fc2 = torch.nn.Linear(l3, l4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        
        val = self.val_relu1(self.val_fc1(x))
        val = self.val_fc2(val)
        
        adv = self.adv_relu1(self.adv_fc1(x))
        adv = self.adv_fc2(adv)
        
        # Combine
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

def train_agent(variant='basic', epochs=2000):
    print(f"--- Training {variant} DQN for player mode ---")
    if variant == 'dueling':
        model = DuelingDQN()
    else:
        model = BasicDQN()
        
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    losses = []
    win_rates = []
    
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    
    sync_freq = 500 # for double DQN
    j = 0 # step counter
    
    if variant == 'double':
        target_model = copy.deepcopy(model)
        target_model.eval()

    wins = 0
    for i in range(epochs):
        game = Gridworld(size=5, mode='player')
        state1_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
            state2 = torch.from_numpy(state2_).float()

            reward = game.reward()
            done = True if reward > 0 or reward < -1 else False # goal or pit
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch)
                
                if variant == 'double':
                    with torch.no_grad():
                        Q2_main = model(state2_batch)
                        Q2_target = target_model(state2_batch)
                    best_actions = torch.argmax(Q2_main, dim=1)
                    Q2_max = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze()
                    Y = reward_batch + gamma * ((1 - done_batch) * Q2_max)
                else:
                    with torch.no_grad():
                        Q2 = model(state2_batch)
                    Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
                    
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if variant == 'double' and j % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0
                if reward > 0:
                    wins += 1
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 100 == 0:
            win_rates.append(wins / 100.0)
            wins = 0
        if (i+1) % 500 == 0:
            print(f"Epoch {i+1}/{epochs} completed.")

    return model, losses, win_rates

def test_model(model, start_pos, mode='player', display=False):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    test_game = Gridworld(size=5, mode=mode)
    
    test_game.board.components['Player'].pos = start_pos
    
    state_ = test_game.board.render_np().reshape(1,100) + np.random.rand(1,100)/10.0
    state = torch.from_numpy(state_).float()
    
    path = []
    path.append(test_game.board.components['Player'].pos)
    
    status = 1
    i = 0
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        
        path.append(test_game.board.components['Player'].pos)
        
        state_ = test_game.board.render_np().reshape(1,100) + np.random.rand(1,100)/10.0
        state = torch.from_numpy(state_).float()
        
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
            else:
                status = 0
        i += 1
        if (i > 15):
            break
            
    win = True if status == 2 else False
    return win, path

import matplotlib.patches as patches

def plot_grids(path_double, path_dueling, start_pos):
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    
    goal = (0, 0)
    pit = (0, 1)
    wall = (1, 1)
    
    paths = [(path_double, 'Double DQN Path', 'blue'), 
             (path_dueling, 'Dueling DQN Path', 'purple')]
             
    for idx, ax in enumerate(axes):
        path, title, color = paths[idx]
        
        ax.set_xticks(np.arange(-0.5, 5, 1))
        ax.set_yticks(np.arange(-0.5, 5, 1))
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        for r in range(5):
            for c in range(5):
                rect = None
                text = None
                if (r, c) == goal:
                    rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='lightgreen')
                    text = 'Goal'
                elif (r, c) == pit:
                    rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='salmon')
                    text = 'Pit'
                elif (r, c) == wall:
                    rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='gray')
                    text = 'Wall'
                elif (r, c) == start_pos:
                    rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='lightyellow')
                    text = 'Start'
                
                if rect:
                    ax.add_patch(rect)
                    ax.text(c, 4-r, text, va='center', ha='center', fontsize=12, fontweight='bold')
        
        if path:
            x = [pos[1] for pos in path]
            y = [4 - pos[0] for pos in path]
            
            ax.plot(x, y, marker='o', color=color, linewidth=3, markersize=8, label=title)
            
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig('hw3-2_paths_comparison.jpg', dpi=300, bbox_inches='tight')
    print("Path plot saved as hw3-2_paths_comparison.jpg")

if __name__ == '__main__':
    epochs = 1000
    
    model_double, losses_double, win_rates_double = train_agent('double', epochs=epochs)
    model_dueling, losses_dueling, win_rates_dueling = train_agent('dueling', epochs=epochs)
    
    # Plot Losses (Using moving average to smooth them out)
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(losses_double) > 100:
        plt.plot(moving_average(losses_double), label='Double DQN', alpha=0.7, color='blue')
        plt.plot(moving_average(losses_dueling), label='Dueling DQN', alpha=0.7, color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Moving Avg)')
    plt.title('Loss over Training Steps')
    plt.legend()
    
    # Plot Win Rates
    plt.subplot(1, 2, 2)
    x_epochs = np.arange(100, epochs+1, 100)
    plt.plot(x_epochs, win_rates_double, marker='s', label='Double DQN', color='blue')
    plt.plot(x_epochs, win_rates_dueling, marker='^', label='Dueling DQN', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Win Rate (per 100 epochs)')
    plt.title('Win Rate over Epochs (Player Mode)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw3-2_comparison.png', dpi=300)
    print("Saved comparison plot to hw3-2_comparison.png")

    print("\nTesting trained models on a fixed random start position...")
    # Find a valid start position (not on pit, goal, or wall)
    valid_starts = [(r, c) for r in range(5) for c in range(5) if (r, c) not in [(0,0), (0,1), (1,1)]]
    start_pos = valid_starts[np.random.randint(0, len(valid_starts))]
    print(f"Chosen start position: {start_pos}")
    
    win_double, path_double = test_model(model_double, start_pos, mode='player')
    win_dueling, path_dueling = test_model(model_dueling, start_pos, mode='player')
    
    print(f"Double DQN won: {win_double}")
    print(f"Dueling DQN won: {win_dueling}")
    
    plot_grids(path_double, path_dueling, start_pos)
