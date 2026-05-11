# 深度強化學習 HW3 規格書 (OpenSpec)

## 1. HW3-1: Naive DQN for static mode [30%]
- **目標**: 實作並執行基本的 DQN (Deep Q-Network) 來解決靜態模式 (static mode) 的簡單環境。
- **要求**:
  - 執行提供的 Naive DQN 程式碼或包含經驗回放緩衝區 (Experience Replay Buffer) 的程式碼。
  - 與 ChatGPT 對話以釐清對程式碼的理解。
  - 提交一份簡短的理解報告 (understanding report)。
- **包含內容**: 
  - 適用於簡單環境的基礎 DQN 實作。
  - Experience Replay Buffer。

## 2. HW3-2: Enhanced DQN Variants for player mode [40%]
- **目標**: 實作並比較進階的 DQN 變體，應用於玩家模式 (player mode)。
- **實作內容**:
  - Double DQN
  - Dueling DQN
- **重點**: 著重於分析這些變體如何改善基礎 DQN 的表現與缺點。

## 3. HW3-3: Enhance DQN for random mode WITH Training Tips [30%]
- **目標**: 增強 DQN 以解決隨機模式 (random mode)，並加入訓練技巧。
- **要求**:
  - 將原本的 PyTorch DQN 模型轉換為 Keras 或 PyTorch Lightning。
  - **加分項目**: 整合訓練技巧以穩定或加速學習過程 (例如：梯度裁剪 gradient clipping、學習率排程 learning rate scheduling 等)。

## 4. HW3-4: Rainbow DQN
- **目標**: 使用 Rainbow DQN 來解決 Random Mode GridWorld。
- **要求**: 先進行分析，然後實作。

## 5. 參考資料 (Reference)
- 本次作業基於 GitHub 儲存庫: [Deep Reinforcement Learning in Action](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)
