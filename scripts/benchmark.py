import torch
import cv2
import time
import numpy as np
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointNet

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

net = SuperPointNet()
net.load_state_dict(torch.load('SuperPointPretrainedNetwork/superpoint_v1.pth',
                               map_location=device))
net.eval()
net.to(device)

# preprocess image
image_path = '../data/imageA.png'  # 替换成你自己的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

image = image.astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

# warm up
warm_up_runs = 5
with torch.no_grad():
    for _ in range(warm_up_runs):
        _ = net(image_tensor)

# iterate 
num_runs = 1000
start_time = time.time()

with torch.no_grad():
    for _ in range(num_runs):
        _ = net(image_tensor)

end_time = time.time()
avg_time_per_run = (end_time - start_time) / num_runs

print(f"Total time for {num_runs} runs: {end_time - start_time:.4f} seconds")
print(f"Average time per run: {avg_time_per_run:.6f} seconds")