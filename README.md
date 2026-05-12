# HW3 — Deep Reinforcement Learning on GridWorld

基於 [Deep Reinforcement Learning in Action](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master) 實作的 DQN 系列作業，環境為 5×5 的 GridWorld。

---

## 作業結構

```
HW3/
├── README.md
├── hw3-1/
│   ├── hw3-1.py
│   └── hw3_1_report.pdf
├── hw3-2/
│   ├── hw3-2.py
│   └── hw3_2_report.pdf
├── hw3-3/
│   ├── hw3-3.py
│   └── hw3_3_report.pdf
└── hw3-4/
    ├── hw3-4.py
    └── hw3_4_report.pdf
```

---

## HW3-1 — Naive DQN for Static Mode［30%］

實作基本的 DQN（Deep Q-Network）解決固定地圖（static mode）的 GridWorld。

**技術重點**
- 三層全連接神經網路（100 → 150 → 100 → 4）
- Experience Replay Buffer（容量 1000，batch size 200）打破時序相關性，穩定訓練

**結果**：1000 epochs 內 loss 穩定收斂，win rate 達 98%+。

---

## HW3-2 — Enhanced DQN Variants for Player Mode［40%］

在起點隨機的 player mode 下，實作並比較兩種進階 DQN 變體。

**實作內容**
- **Double DQN**：將動作選擇（main network）與價值評估（target network）拆開，解決過度估計（overestimation）問題
- **Dueling DQN**：將網路分流為 Value Stream V(s) 與 Advantage Stream A(s,a)，提升狀態泛化能力

**結果**：Dueling DQN 在前 100 epochs 勝率即達 92%，優於 Double DQN 的 67%；兩者最終皆收斂至近 100%。

---

## HW3-3 — Enhanced DQN for Random Mode with Training Tips［30%+加分］

將 PyTorch DQN 移植至高階框架，並加入訓練技巧應對完全隨機地圖（random mode）。

**實作內容**
- **Keras**：`@tf.function` 圖編譯加速，`tf.GradientTape` 精確控制梯度
- **PyTorch Lightning**：`LightningModule` + `IterableDataset` 模組化架構

**加分項目（均已實作）**
- Gradient Clipping：`clipnorm=1.0`（Keras）/ `gradient_clip_val=1.0`（Lightning），防止梯度爆炸
- Learning Rate Scheduling：`ExponentialDecay`（Keras）/ `StepLR`（Lightning），後期穩定收斂

**結果**：3000 epochs 後，Lightning 勝率達 75~80%，Keras 達 65~70%。

---

## HW3-4 — Rainbow DQN for Random Mode［加分挑戰］

整合六項頂尖技術的 Rainbow DQN，解決最複雜的 random mode。

**實作組件**
| 組件 | 解決的問題 |
|---|---|
| Distributional RL (C51) | 預測回報機率分佈而非單一期望值，掌握風險 |
| NoisyNets | 以參數化雜訊取代 ε-greedy，實現自適應探索 |
| Prioritized Experience Replay (PER) | 優先學習 TD Error 大的高價值經驗 |
| N-step Returns | 加速獎勵傳遞，改善 Sparse Reward 問題 |
| Dueling Network | 分離狀態價值與動作優勢，提升學習效率 |
| Double DQN | 解決 Q 值過度估計 |

**超參數調整**：`std_init=0.1`（避免小地圖過度探索）、`v_min=-50, v_max=50`（對應最多 50 步懲罰）、訓練 20000 epochs。

**結果**：Loss 從 3.5 平滑下降至趨近 0；測試時關閉 NoisyNets 後，模型能對任意隨機地圖規劃出最短路徑。

---

## 環境需求

```bash
pip install torch pytorch-lightning tensorflow numpy matplotlib
```

GridWorld 環境來自：[DeepReinforcementLearningInAction](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)，執行前請將 `Gridworld.py` 放在同一目錄下。
