import cv2
import numpy as np

# Load the images
img1 = cv2.imread('/Users/taimoorrizwan/Downloads/PURE/pure/ForPublication/01-01/Image1392643995576149000.png')
img2 = cv2.imread('/Users/taimoorrizwan/Downloads/PURE/01-01/01-01/Image1392643995542835000.png')

# Stack the images into a 4D array
data = np.stack((img2, img1))
def diff_normalize_data(data):
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len - 1):
        diff = data[j + 1, :, :, :] - data[j, :, :, :]
        sum = data[j + 1, :, :, :] + data[j, :, :, :]
        diffnormalized_data[j, :, :, :] = np.divide(diff, sum + 1e-7, out=np.zeros_like(diff), where=(sum != 0))
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    std = np.std(diffnormalized_data)
    if std != 0:
        diffnormalized_data = diffnormalized_data / std
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    return diffnormalized_data
# Apply differential normalization
diffnormalized_data = diff_normalize_data(data)

# Normalize the pixel values to 0-255
diffnormalized_data = (diffnormalized_data * 255).astype(np.uint8)

# Save the resulting image
cv2.imwrite('/Users/taimoorrizwan/Downloads/image3.jpg', diffnormalized_data[0])