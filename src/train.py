import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ShallowUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'telemetry_log_2026_06_to_08.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'unet_segmentation_weights.pth')

class MicroGridDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def extract_and_reshape_data():
    print("[PRE-PROCESS] Ingesting telemetry database from CSV...")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set targets: Minimal (0-5ppb, Class 0), Low (5-10ppb, Class 1), Medium (10-20ppb, Class 2), High (>20ppb, Class 3)
    df['target'] = pd.cut(df['methane_anomaly'].fillna(1.95), bins=[-np.inf, 5, 10, 20, np.inf], labels=[0, 1, 2, 3]).astype(int)
    
    # Sort to enforce grid-structure
    df = df.sort_values(by=['timestamp', 'field_name', 'sector_row', 'sector_col'])
    
    features = ['ndvi', 'ndwi', 'savi', 'lst', 'soil_moisture']
    
    # Reshape features to (N, 5, 8, 8)
    grouped = df.groupby(['timestamp', 'field_name'])
    
    inputs_list, targets_list, dates_list = [], [], []
    for (t_val, f_name), group in grouped:
        if len(group) == 64:
            x = group[features].values.reshape(8, 8, 5).transpose(2, 0, 1)
            y = group['target'].values.reshape(8, 8)
            inputs_list.append(x)
            targets_list.append(y)
            dates_list.append(t_val)
            
    print(f"[PRE-PROCESS] Extracted {len(inputs_list)} complete 8x8 grids in {time.time()-t0:.2f}s.")
    return np.array(inputs_list), np.array(targets_list), dates_list

def main():
    inputs, targets, dates = extract_and_reshape_data()
    dates_pd = pd.to_datetime(dates)
    
    # Split by month: June & July for training, August for testing
    train_mask = dates_pd.month.isin([6, 7])
    test_mask = dates_pd.month == 8
    
    x_train, y_train = inputs[train_mask], targets[train_mask]
    x_test, y_test = inputs[test_mask], targets[test_mask]
    
    # Inject 15% Gaussian noise into train set
    noise = np.random.normal(0, 0.15, size=x_train.shape)
    x_train_noisy = x_train + noise
    
    print(f"[SPLIT] Train: {len(x_train_noisy)} grids (June/July, 15% noise)")
    print(f"[SPLIT] Test: {len(x_test)} grids (August, clean)")
    
    train_dataset = MicroGridDataset(x_train_noisy, y_train)
    test_dataset = MicroGridDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)
    
    model = ShallowUNet(in_channels=5, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    print("[TRAINING] Starting 20 Epoch Optimization Loop...")
    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_dataset)
        
        # Test Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()
                
        val_acc = (correct / total) * 100
        print(f"  Epoch {epoch:02d}/20 | Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
    # Save final model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[SUCCESS] Weights saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()
