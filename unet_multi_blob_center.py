import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
import time

class BlobDataset(Dataset):
    def __init__(self, images, center_coords_list, sigma=0.5):
        self.images = images.astype(np.float32)
        self.coord_lists = center_coords_list
        self.sigma = sigma

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx][None, :, :]
        y = self._generate_multi_blob_heatmap(self.coord_lists[idx])
        return torch.tensor(x), torch.tensor(y)

    def _generate_multi_blob_heatmap(self, centers):
        h, w = 64, 64
        heatmap = np.zeros((h, w), dtype=np.float32)
        for x, y in centers:
            if 0 <= x < w and 0 <= y < h:
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                heatmap += np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap[None, :, :]

## The Neural Network class
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ## Dropout, recommended is a value of 0.2, 0.25 leads to poorer results
                nn.Dropout(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ## Another dropout.
                nn.Dropout(0.2) 
            )

        ## Downsampling with 6 layers
        self.enc1 = conv_block(1, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.enc5 = conv_block(256, 512)
        self.enc6 = conv_block(512, 1024)

        ## Dialated bottleneck layer
        self.bottleneck = conv_block(1024, 2048)
        self.dilated = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        ## Upsampling with 6 layers.
        self.dec6 = conv_block(2048 + 1024, 1024)
        self.dec5 = conv_block(1024 + 512, 512)
        self.dec4 = conv_block(512 + 256, 256)
        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.dec1 = conv_block(64 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

        ## Model technically looks like this: \_/

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))
        e6 = self.enc6(F.max_pool2d(e5, 2))

        b = self.bottleneck(F.max_pool2d(e6, 2))
        b = self.dilated(b)

        d6 = self.dec6(torch.cat([F.interpolate(b, scale_factor=2), e6], dim=1))
        d5 = self.dec5(torch.cat([F.interpolate(d6, scale_factor=2), e5], dim=1))
        d4 = self.dec4(torch.cat([F.interpolate(d5, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        return torch.sigmoid(self.final(d1))

## Non essential function used for post-training eval with training data
def extract_subpixel_peaks(heatmap, threshold=0.001, size=3, window=3):
    assert window % 2 == 1
    h, w = heatmap.shape
    offset = window // 2

    local_max = maximum_filter(heatmap, size=size)
    peak_mask = (heatmap == local_max) & (heatmap > threshold)
    peak_coords = np.argwhere(peak_mask)

    refined_coords = []
    for y, x in peak_coords:
        x_min = max(x - offset, 0)
        x_max = min(x + offset + 1, w)
        y_min = max(y - offset, 0)
        y_max = min(y + offset + 1, h)

        window_vals = heatmap[y_min:y_max, x_min:x_max]
        if window_vals.sum() == 0:
            continue

        xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        cx = (xx * window_vals).sum() / window_vals.sum()
        cy = (yy * window_vals).sum() / window_vals.sum()
        refined_coords.append((cx, cy))

    return refined_coords

## Another non essential function
def peak_matching_loss(pred_peaks, true_peaks):
    if len(pred_peaks) == 0 and len(true_peaks) == 0 :
        return 0.0
    elif len(pred_peaks) == 0 or len(true_peaks) == 0 :
        return float(max(len(pred_peaks), len(true_peaks)))  # total mismatch penalty

    cost_matrix = np.zeros((len(pred_peaks), len(true_peaks)))
    for i, (px, py) in enumerate(pred_peaks):
        for j, (tx, ty) in enumerate(true_peaks):
            cost_matrix[i, j] = np.hypot(px - tx, py - ty)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match_distances = cost_matrix[row_ind, col_ind]
    unmatched = abs(len(pred_peaks) - len(true_peaks))
    return float(match_distances.sum() + unmatched)  # localization + count error


# --- Hybrid Loss Function ---
def hybrid_loss(pred, target):
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    return bce(pred, target) + mse(pred, target)


# --- Training Routine ---
def train_model(X, Y_coords, epochs=25, batch_size=32, lr=5e-4):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_coords, test_size=0.2, random_state=42)
    train_ds = BlobDataset(X_train, Y_train)
    val_ds = BlobDataset(X_val, Y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (xb, yb) in enumerate(train_dl) :
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = hybrid_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_dl):.4f}")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Local Evaluation, comment this out as this is testing on training data, not nessecarily applicable to actual performance.
    model.eval()
    for i in range(3):
        xb, _ = val_ds[i]
        with torch.no_grad():
            pred = model(xb[None].to(device)).cpu().squeeze().numpy()
        pred_peaks = extract_subpixel_peaks(pred, threshold=0.01)
        gt_peaks = Y_val[i]

        match_loss = peak_matching_loss(pred_peaks, gt_peaks)
        print(f"Eval Sample {i}: Localization + Count Loss = {match_loss:.4f}")

        plt.figure()
        plt.imshow(X_val[i], cmap='hot')
        if len(gt_peaks) > 0:
            xs, ys = zip(*gt_peaks)
            plt.scatter(xs, ys, c='blue', label='True', s=50)
        if len(pred_peaks) > 0:
            px, py = zip(*pred_peaks)
            plt.scatter(px, py, c='lime', label='Predicted', marker='x')
        plt.title("Blob Center Prediction (Improved)")
        plt.legend()
        plt.show()

    ## Returns model back to runner
    return model