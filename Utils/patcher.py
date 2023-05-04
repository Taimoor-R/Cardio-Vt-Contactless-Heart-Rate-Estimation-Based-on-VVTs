import numpy as np
from patchify import patchify
from PIL import Image

image = Image.open("/Users/taimoorrizwan/Downloads/subj_1001_idx_13.png")  # for example (3456, 5184, 3)
image = np.asarray(image)
patches = patchify(image, (8, 8, 3), step=1)
print(patches.shape)  # (6, 10, 1, 512, 512, 3)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]
        patch = Image.fromarray(patch)
        num = i * patches.shape[1] + j
        patch.save(f"/Users/taimoorrizwan/Downloads/patch_{num}.jpg")