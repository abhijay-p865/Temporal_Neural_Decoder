import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CNN import THREE_DIMENSIONAL_CNN
from function_calls import eval_acc_auc

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


data_dir = r"C:\Users\abhij\OneDrive\Desktop\Replicate #2\data\surface_d5_p0.001_k29"


X = np.load(os.path.join(data_dir, "X.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))


N = X.shape[0]

indices = np.arange(N)
np.random.shuffle(indices)

split = int(0.8 * N)
train_idx = indices[:split]
test_idx  = indices[split:]


X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print("test label rate:", float(y_test.mean()))
print("majority-class baseline acc:", float(baseline_acc))
print("train samples:", X_train.shape[0])
print("test samples:", X_test.shape[0])

class ReplicateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = ReplicateDataset(X_train, y_train)
test_dataset  = ReplicateDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: " + str(device))


channels = X.shape[1]
time_depth = X.shape[2]
x_dim = X.shape[3]
y_dim = X.shape[4]

print("detected dimensions:", channels, time_depth, x_dim, y_dim)

model = THREE_DIMENSIONAL_CNN(channels, time_depth, x_dim, y_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model.train()
num_epochs = 15

best_auc = -1.0
best_epoch = -1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs).squeeze()   # (batch,)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval()
    test_acc, test_auc = eval_acc_auc(model,test_loader,device)    
    if test_auc>best_auc:
        best_auc=test_auc
        best_epoch = epoch +1
        torch.save(model.state_dict(), os.path.join(data_dir, "best_model.pt"))
    print(
    f"Epoch {epoch+1} | "
    f"loss {avg_loss:.4f} | "
    f"test acc {test_acc:.3f} | "
    f"test auc {test_auc:.3f}"
)

torch.save(model.state_dict(), os.path.join(data_dir, "model_k10.pt"))
print("saved model")
