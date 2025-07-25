import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

for root, dirs, files in os.walk('../datasets/jmsd/perspective_v1/masks'):
    print(root)
    for file in tqdm(files):
        if file.endswith('.png') and not file.endswith('.jpg.png'):
            file_path = os.path.join(root, file)
            img = Image.open(file_path)
            img_array = np.array(img)
            inverted_array = 255 - img_array
            inverted_img = Image.fromarray(inverted_array.astype(np.uint8))
            inverted_img.save(file_path.replace('.png', '.jpg.png'))
            os.remove(file_path)