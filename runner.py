import numpy as np  
from unet_multi_blob_center import train_model
import torch

def read_data_file(filepath):
    true_clusters_coords = []
    images = []

    with open(filepath, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        num_coords = int(lines[i].strip())
        i += 1
        coords = []
        for _ in range(num_coords):
            y, x = map(float, lines[i].strip().split())
            coords.append([y, x])
            i += 1
        true_clusters_coords.append(np.array(coords)[:, ::-1]) ## Flips it from Y, X to X, Y

        grid = []
        for _ in range(64):
            row = list(map(int, lines[i].strip().split()))
            grid.append(row)
            i += 1
        images.append(np.array(grid))

    return true_clusters_coords, images

if __name__ == "__main__" :
    ## Due to the way that our intitial data is formatted
    ## Thed ata file must be formatted as 
    ## number of clusters
    ## clusters coordinates in Y, X
    ## The 64 by 64 heatmap image
    ## It is important to note that the Y representes time 
    ## and the X repesents the X coordinate. This is because
    ## of time projection chambers' unique data readout.

    cluster_coords, image_array = read_data_file("shuffled_high_density_training_file.txt")

    X = np.array(image_array, dtype=np.float32) 
    Y_coords = cluster_coords  
    
    epochs = 100 ## Change this to your value, 90-110 yields the best results generally.
    model = train_model(X, Y_coords, epochs)

    torch.save(model.state_dict(), f"unet_blob_detector_{str(epochs)}_epoch.pth")