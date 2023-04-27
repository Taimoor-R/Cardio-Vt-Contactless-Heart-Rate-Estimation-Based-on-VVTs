import cv2
import numpy as np
import math
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import os
from dataset.metrics import predict_calculate_metrics
from timesformer_pytorch.timesformer_pytorch import TimeSformer
from torchvision.transforms import ToTensor, Resize, CenterCrop

# Load the npy file
data = np.load('/notebooks/TimeSformer-pytorch/dataset/lucy_56_0_tensors.npy')

# Preprocess the data
# ... (same preprocessing code used before saving to npy file)

data_tensor = torch.from_numpy(data).float()#.unsqueeze(0).permute(0, 1, 4, 2, 3)
print(data_tensor.shape)  # torch.Size([1, 32, 3, 64, 64])
# Define the model
print(data_tensor.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the trained model
model = TimeSformer(
    dim=512,
    image_size=64,
    patch_size=8,
    num_frames=32,
    num_classes=32,
    depth=32,
    heads=8,
    dim_head=64,
    attn_dropout=0.2,
    ff_dropout=0.2
).to(device)
model.load_state_dict(torch.load('/notebooks/rPPG-Toolbox/PreTrainedModels/PURE_SizeW64_SizeH64_ClipLength32_DataTypeDiffNormalized_LabelTypeDiffNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len128/PURE_PURE_PURE_physnet_diffnormalized_Epoch20.pth'))
model.eval()
output_tensor = torch.zeros_like(data_tensor)

# Make predictions for each batch
# Make predictions on the video
with torch.no_grad():
    model.eval()
    predictions = []
    for batch in data_tensor:
        batch = batch.unsqueeze(0)
        batch = batch.to('cuda')
        prediction = model(batch)
        predictions.append(prediction.cpu().numpy())

# Combine predictions from each batch
predictions = np.concatenate(predictions)
values = predictions.flatten().tolist()
#print(values)
time = np.arange(len(values))
plt.plot(time, values)
plt.xlabel('Time (s)')
plt.ylabel('PPG Amplitude')
plt.title('PPG Graph')
plt.savefig('plot.png')

# assuming predicted_ppg is a numpy array of shape (T,)
predict_calculate_metrics(values)