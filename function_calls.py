import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

#function to build the spacetime grid from the simulation output
def build_spacetime(events_coords, det_values, keep_t=(1, 29)):
    X = 6
    Y = 6

    t_min, t_max = keep_t
    T = t_max - t_min + 1  # e.g. 29 rounds if 1..29

    events = np.zeros((T, X, Y), dtype=np.float32)
    mask   = np.zeros((T, X, Y), dtype=np.float32)

    for det_id, (x, y, t) in events_coords.items():
        t = int(t)
        if t < t_min or t > t_max:
            continue

        gx = int(x // 2)  # 0..5
        gy = int(y // 2)  # 0..5
        tt = t - t_min     # re-index time to start at 0

        events[tt, gx, gy] = float(det_values[det_id])
        mask[tt, gx, gy]   = 1.0

    return events, mask

#function to shape the simulation output into the input for the CNN
def make_cnn_input(events, mask, k):
    """
    events, mask: (T, 6, 6)
    returns: (2, k, 6, 6)
    """
    ev = events[-k:]  # last k time slices
    mk = mask[-k:]
    x = np.stack([ev, mk], axis=0)  # channels first
    return x.astype(np.float32)


#function to evaluate accuracy and AUC of the model
def eval_acc_auc(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb).squeeze()
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs)
    acc = correct / total
    return acc, auc