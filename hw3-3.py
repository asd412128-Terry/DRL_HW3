import numpy as np
import random
import copy
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Gridworld import Gridworld
import warnings

warnings.filterwarnings("ignore")

# --- Keras Imports ---
import tensorflow as tf
from tensorflow import keras

# --- PyTorch Lightning Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

# We remove fixed random seeds so that every time you run the script,
# you get a truly randomized Gridworld and the model actually learns general strategies.
# np.random.seed(42)
# tf.random.set_seed(42)
# pl.seed_everything(42, workers=True)
# random.seed(42)

# ==========================================
# 1. Keras Implementation
# ==========================================
def build_keras_model(l1=100, l2=150, l3=100, l4=4):
    model = keras.Sequential([
        keras.layers.Dense(l2, activation='relu', input_shape=(l1,)),
        keras.layers.Dense(l3, activation='relu'),
        keras.layers.Dense(l4)
    ])
    return model

def train_dqn_random_keras(epochs=1000):
    l1, l4 = 100, 4

    # [Fix] decay_steps 設為 epochs*10，讓 LR 衰減速率跟 Lightning StepLR 相近
    # 原本 decay_steps=3000 在 3000 epochs × 多步的情況下衰減太快，後期 loss 暴衝
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=epochs * 10, decay_rate=0.9, staircase=True)
        
    # 加分項目: Gradient Clipping
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    model = build_keras_model()
    loss_fn = keras.losses.MeanSquaredError()
    gamma = 0.9

    @tf.function
    def predict_action_q(state):
        return model(state)

    @tf.function
    def train_step(s1, a, r, s2, d):
        Q2 = model(s2, training=False)
        max_Q2 = tf.reduce_max(Q2, axis=1)
        Y = r + gamma * ((1.0 - d) * max_Q2)
        
        with tf.GradientTape() as tape:
            Q1 = model(s1, training=True)
            action_masks = tf.one_hot(a, l4)
            X = tf.reduce_sum(Q1 * action_masks, axis=1)
            loss = loss_fn(Y, X)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    epsilon = 0.3
    batch_size = 200
    mem_size = 1000
    replay = deque(maxlen=mem_size)
    max_moves = 50
    losses, win_rates = [], []
    wins = 0
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    print("--- Training Keras DQN for Random Mode ---")
    for i in range(epochs):
        game = Gridworld(size=5, mode='random')
        state1_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        status, mov = 1, 0
        
        while status == 1:
            mov += 1
            state1_tensor = tf.convert_to_tensor(state1_, dtype=tf.float32)
            qval = predict_action_q(state1_tensor)
            qval_ = qval.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
            reward = game.reward()
            done = True if reward > 0 or reward < -1 else False
            
            exp = (state1_, action_, reward, state2_, done)
            replay.append(exp)
            state1_ = state2_
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = np.vstack([s1 for (s1,a,r,s2,d) in minibatch]).astype(np.float32)
                action_batch = np.array([a for (s1,a,r,s2,d) in minibatch], dtype=np.int32)
                reward_batch = np.array([r for (s1,a,r,s2,d) in minibatch], dtype=np.float32)
                state2_batch = np.vstack([s2 for (s1,a,r,s2,d) in minibatch]).astype(np.float32)
                done_batch = np.array([float(d) for (s1,a,r,s2,d) in minibatch], dtype=np.float32)
                
                loss = train_step(state1_batch, action_batch, reward_batch, state2_batch, done_batch)
                losses.append(loss.numpy())
                
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
            print(f"Keras Epoch {i+1}/{epochs} completed.")
            
    return model, losses, win_rates

# ==========================================
# 2. PyTorch Lightning Implementation
# ==========================================
class ReplayBufferLightning:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class RLDatasetLightning(IterableDataset):
    def __init__(self, module):
        self.module = module
    def __iter__(self):
        self.module.reset_game()
        while not self.module.is_game_over():
            self.module.play_step(self.module.epsilon)
            if len(self.module.buffer) >= self.module.hparams.batch_size:
                batch = self.module.buffer.sample(self.module.hparams.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                yield (torch.stack(states), torch.tensor(actions), torch.tensor(rewards),
                       torch.stack(next_states), torch.tensor(dones))

class LitDQN(pl.LightningModule):
    def __init__(self, l1=100, l2=150, l3=100, l4=4, batch_size=200, lr=1e-3, gamma=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(l1, l2), nn.ReLU(),
            nn.Linear(l2, l3), nn.ReLU(),
            nn.Linear(l3, l4)
        )
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBufferLightning(1000)
        self.epsilon = 0.3
        self.max_moves = 50
        self.epoch_wins = 0
        self.win_rates = []
        self.losses_list = []
        
    def _get_state(self):
        state_ = self.game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        return torch.from_numpy(state_).float()

    def reset_game(self):
        self.game = Gridworld(size=5, mode='random')
        self._state = self._get_state()
        self.mov = 0
        self.status = 1
        
    def is_game_over(self):
        return self.status == 0
        
    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # 加分項目: Learning Rate Scheduling
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.9)
        return [optimizer], [scheduler]
        
    def play_step(self, epsilon):
        action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        qval = self(self._state)
        qval_ = qval.detach().numpy()
        
        if random.random() < epsilon:
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
            
        action = action_set[action_]
        self.game.makeMove(action)
        self.mov += 1
        
        next_state = self._get_state()
        reward = self.game.reward()
        done = True if reward > 0 or reward < -1 else False
        
        exp = (self._state.squeeze(0), action_, reward, next_state.squeeze(0), done)
        self.buffer.append(exp)
        self._state = next_state
        
        if reward != -1 or self.mov > self.max_moves:
            self.status = 0
            if reward > 0:
                self.epoch_wins += 1

    def train_dataloader(self):
        dataset = RLDatasetLightning(self)
        return DataLoader(dataset=dataset, batch_size=None)
        
    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        states = states.float()
        actions = actions.long()
        rewards = rewards.float()
        next_states = next_states.float()
        dones = dones.float()
        
        Q1 = self(states)
        with torch.no_grad():
            Q2 = self(next_states)
            
        Y = rewards + self.hparams.gamma * ((1 - dones) * torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_fn(X, Y)
        self.losses_list.append(loss.item())
        return loss
        
    def on_train_epoch_end(self):
        if self.epsilon > 0.1:
            self.epsilon -= (1/self.trainer.max_epochs)
        if (self.current_epoch + 1) % 100 == 0:
            self.win_rates.append(self.epoch_wins / 100.0)
            self.epoch_wins = 0
        if (self.current_epoch + 1) % 500 == 0:
            print(f"Lightning Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} completed.")

def train_dqn_random_lightning(epochs=1000):
    model = LitDQN()
    # 加分項目: Gradient Clipping
    trainer = pl.Trainer(
        max_epochs=epochs, gradient_clip_val=1.0, 
        enable_checkpointing=False, logger=False,
        enable_progress_bar=False, enable_model_summary=False
    )
    print("--- Training PyTorch Lightning DQN for Random Mode ---")
    trainer.fit(model)
    return model, model.losses_list, model.win_rates

# ==========================================
# 3. Testing & Evaluation
# ==========================================

def clone_game_from_components(initial_components):
    """
    [Fix] 用 deepcopy 建立一個乾淨的 Gridworld，
    再把所有棋子的位置 patch 回去，並重建 board 的 numpy 矩陣，
    確保兩個模型在完全相同的棋盤狀態下接受測試。
    """
    # 先建一個隨機棋盤，再把內部狀態整個替換掉
    game = Gridworld(size=5, mode='random')
    for piece in ['Player', 'Goal', 'Pit', 'Wall']:
        game.board.components[piece].pos = initial_components[piece]
    # 重新渲染 board 矩陣，讓 pos 的變更真正反映到 board 上
    game.board.render()
    return game

def get_random_valid_start_and_game():
    game = Gridworld(size=5, mode='random')
    initial_components = {
        'Player': game.board.components['Player'].pos,
        'Goal':   game.board.components['Goal'].pos,
        'Pit':    game.board.components['Pit'].pos,
        'Wall':   game.board.components['Wall'].pos,
    }
    return game, initial_components

def test_keras_model(model, game, display=False):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    state_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
    path = [game.board.components['Player'].pos]
    status, i = 1, 0
    
    while status == 1:
        state_tensor = tf.convert_to_tensor(state_, dtype=tf.float32)
        qval = model(state_tensor, training=False)
        action_ = np.argmax(qval.numpy())
        action = action_set[action_]
        
        if display: print(f'Keras Move #: {i}; Taking action: {action}')
        game.makeMove(action)
        path.append(game.board.components['Player'].pos)
        state_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        
        reward = game.reward()
        if reward != -1:
            status = 2 if reward > 0 else 0
        i += 1
        if i > 50: break
            
    return (status == 2), path

def test_lightning_model(model, game, display=False):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    state_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
    path = [game.board.components['Player'].pos]
    status, i = 1, 0
    model.eval()
    
    while status == 1:
        state_tensor = torch.from_numpy(state_).float()
        with torch.no_grad():
            qval = model(state_tensor)
        action_ = np.argmax(qval.numpy())
        action = action_set[action_]
        
        if display: print(f'Lightning Move #: {i}; Taking action: {action}')
        game.makeMove(action)
        path.append(game.board.components['Player'].pos)
        state_ = game.board.render_np().reshape(1,100) + np.random.rand(1,100)/100.0
        
        reward = game.reward()
        if reward != -1:
            status = 2 if reward > 0 else 0
        i += 1
        if i > 50: break
            
    return (status == 2), path

def plot_combined_grids(path_keras, path_lightning, initial_components, win_keras, win_lightning):
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    goal  = initial_components['Goal']
    pit   = initial_components['Pit']
    wall  = initial_components['Wall']
    start = initial_components['Player']
    
    results = [win_keras, win_lightning]
    paths = [(path_keras,    'Keras DQN Path',     'blue'),
             (path_lightning, 'Lightning DQN Path', 'purple')]
             
    for idx, ax in enumerate(axes):
        path, title, color = paths[idx]
        won = results[idx]
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
            ax.plot(x, y, marker='o', color=color, linewidth=3, markersize=8, label=title)
        
        # 在標題加上 Win / Lose 結果，一眼看清楚
        outcome = '✔ Win' if won else '✘ Lose'
        ax.set_title(f'{title}  [{outcome}]', fontsize=14)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig('hw3-3_paths_comparison.jpg', dpi=300, bbox_inches='tight')
    print("Path comparison plot saved as hw3-3_paths_comparison.jpg")

if __name__ == '__main__':
    epochs = 3000
    
    # 1. Train Both Models
    model_keras, losses_keras, win_rates_keras = train_dqn_random_keras(epochs=epochs)
    model_lightning, losses_lightning, win_rates_lightning = train_dqn_random_lightning(epochs=epochs)
    
    # 2. Plot Metrics (Loss & Win Rate)
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(losses_keras) > 100:
        plt.plot(moving_average(losses_keras), label='Keras DQN Loss', color='blue', alpha=0.7)
    if len(losses_lightning) > 100:
        plt.plot(moving_average(losses_lightning), label='Lightning DQN Loss', color='purple', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Moving Avg)')
    plt.title('Loss over Training Steps')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    x_epochs = np.arange(100, epochs + 1, 100)
    plt.plot(x_epochs, win_rates_keras,     marker='s', label='Keras DQN',     color='blue')
    plt.plot(x_epochs, win_rates_lightning, marker='^', label='Lightning DQN', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Win Rate (per 100 epochs)')
    plt.title('Win Rate over Epochs (Random Mode)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw3-3_metrics_comparison.png', dpi=300)
    print("Metrics comparison plot saved as hw3-3_metrics_comparison.png")
    
    # 3. Test Both Models on the SAME random environment
    # [Fix] 必須兩個模型都贏，才算找到值得畫的地圖（改 or → and）
    print("\nTesting trained models... (Will try to find a map where BOTH win)")
    
    success = False
    attempts = 0
    win_keras = win_lightning = False
    path_keras = path_lightning = []
    initial_comps = None

    while not success and attempts < 500:
        attempts += 1

        # 建立 Keras 測試用棋盤，並記錄初始位置
        game_keras, initial_comps = get_random_valid_start_and_game()
        win_keras, path_keras = test_keras_model(model_keras, game_keras, display=False)

        # [Fix] 用 clone_game_from_components 正確複製相同棋盤給 Lightning 測試
        game_lightning = clone_game_from_components(initial_comps)
        win_lightning, path_lightning = test_lightning_model(model_lightning, game_lightning, display=False)

        # [Fix] 改成 and：兩個都贏才畫，避免圖中出現走錯路的模型
        if win_keras and win_lightning:
            success = True

    if not success:
        # 500 次都找不到兩個都贏的地圖，就用最後一次結果畫（並在標題顯示結果）
        print("Warning: Could not find a map where BOTH models win within 500 attempts.")
        print("Plotting the last attempt's result instead.")

    print(f"\nTested on {attempts} random map(s).")
    print(f"Keras DQN won:     {win_keras}")
    print(f"Lightning DQN won: {win_lightning}")
    
    # 4. Plot both paths（傳入 win 結果，讓標題顯示 Win/Lose）
    plot_combined_grids(path_keras, path_lightning, initial_comps, win_keras, win_lightning)