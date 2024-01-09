import numpy as np
from PIL import Image
import os
from multiprocessing import Pool
from tqdm import tqdm

def process_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1范围
    return img_array.shape[0] * img_array.shape[1], np.sum(img_array, axis=(0, 1)), np.sum(np.square(img_array), axis=(0, 1))

def compute_stats(folders):
    image_paths = []
    for folder in folders:
        image_paths.extend([os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']])
    
    with Pool(processes=os.cpu_count()) as pool:
        stats = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths)))

    # Sum up the statistics returned by process_image
    pixel_count = 0
    sum_pixels = np.zeros(3)
    sum_pixels_squared = np.zeros(3)
    
    for count, pixels, pixels_squared in stats:
        pixel_count += count
        sum_pixels += pixels
        sum_pixels_squared += pixels_squared
    
    # Calculate means and stds
    means = sum_pixels / pixel_count
    stds = np.sqrt(sum_pixels_squared / pixel_count - np.square(means))
    
    return means, stds

if __name__ == '__main__':
    folders_to_include = ['./data/train', './data/test']
    
    overall_means, overall_stds = compute_stats(folders_to_include)
    
    print(f"Overall Means: {overall_means}, Overall Stds: {overall_stds}")

    '''
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37500/37500 [00:32<00:00, 1168.24it/s]
    Overall Means: [0.48611028 0.4537106  0.41540866], Overall Stds: [0.26209406 0.25534465 0.25821651]
    '''