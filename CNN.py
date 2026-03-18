import torch
import torch.nn as nn
import torch.nn.functional as F

class THREE_DIMENSIONAL_CNN(nn.Module):
    def __init__(self, channels, time_depth, x_dim, y_dim):
        super().__init__()
        
        self.channels = channels
        self.time_depth = time_depth
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        #NN layers
        self.Conv1 = nn.Conv3d(channels, 8, kernel_size=(3,3,3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.Conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1)
        self.Conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        # optional dropout (keeps things from overfitting)
        self.dropout = nn.Dropout(0.3)

        # after one spatial pool: 6x6 -> 3x3
        pooled_x = x_dim // 2  # 6 -> 3
        pooled_y = y_dim // 2  # 6 -> 3

        flat = 32 * time_depth * pooled_x * pooled_y
        self.fc1 = nn.Linear(flat, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
      # x: (batch, channels, time, x, y)
      x = F.relu(self.Conv1(x))
      x = self.pool1(x)

      x = F.relu(self.Conv2(x))
      x = F.relu(self.Conv3(x))


      x = x.view(x.size(0), -1)
      x = self.dropout(F.relu(self.fc1(x)))
      x = self.fc2(x).squeeze(1)   # logits
      return x