import numpy as np
import torch
from Gridworld import Gridworld
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def train_dqn_exp_replay():
    l1 = 100
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
    )
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

    epochs = 1000
    losses = []
    win_rates = []
    wins = 0
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    
    print("Training DQN with Experience Replay for static mode...")
    for i in range(epochs):
        game = Gridworld(size = 5, mode='static')
        state1_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        while(status == 1): 
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
            done = True if reward > 0 else False
            exp =  (state1, action_, reward, state2, done)
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
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if reward != -1 or mov > max_moves:
                status = 0
                if reward > 0:
                    wins += 1
                mov = 0
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 100 == 0:
            win_rates.append(wins / 100.0)
            wins = 0
            
        if (i+1) % 500 == 0:
            print(f"Epoch {i+1}/{epochs} completed.")
            
    return model, action_set, losses, win_rates

def test_model(model, action_set, mode='static', display=True):
    i = 0
    test_game = Gridworld(size=5, mode=mode)
    state_ = test_game.board.render_np().reshape(1,100) + np.random.rand(1,100)/10.0
    state = torch.from_numpy(state_).float()
    
    path = []
    # Log initial position
    path.append(test_game.board.components['Player'].pos)
    
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
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
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    
    win = True if status == 2 else False
    return win, path

def plot_grid_and_path(path):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Coordinates for standard setup in mode='static'
    goal = (0, 0)
    pit = (0, 1)
    wall = (1, 1)
    start = (0, 3)
    
    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, 5, 1))
    ax.set_yticks(np.arange(-0.5, 5, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Fill cells
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
            elif (r, c) == start:
                rect = patches.Rectangle((c-0.5, 4-r-0.5), 1, 1, facecolor='lightyellow')
                text = 'Start'
            
            if rect:
                ax.add_patch(rect)
                ax.text(c, 4-r, text, va='center', ha='center', fontsize=12, fontweight='bold')
    
    # Plot path
    if path:
        x = [pos[1] for pos in path]
        y = [4 - pos[0] for pos in path]
        
        ax.plot(x, y, marker='o', color='blue', linewidth=3, markersize=8, label='DQN Path')
        
    ax.set_title('GridWorld DQN Agent Path', fontsize=14)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.legend(loc='upper right')
    
    plt.savefig('hw3-1_path.jpg', dpi=300, bbox_inches='tight')
    print("Path plot saved as hw3-1_path.jpg")

if __name__ == "__main__":
    trained_model, actions, losses, win_rates = train_dqn_exp_replay()
    
    # Plot Metrics
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(losses) > 100:
        plt.plot(moving_average(losses), label='Basic DQN Loss', color='blue')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Moving Avg)')
    plt.title('Loss over Training Steps')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    epochs = 1000
    x_epochs = np.arange(100, epochs+1, 100)
    plt.plot(x_epochs, win_rates, marker='o', label='Basic DQN Win Rate', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Win Rate (per 100 epochs)')
    plt.title('Win Rate over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw3-1_metrics.png', dpi=300)
    print("Metrics plot saved as hw3-1_metrics.png")

    print("\nTesting trained model...")
    win, path = test_model(trained_model, actions, mode='static')
    plot_grid_and_path(path)
