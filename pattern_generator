from PIL import Image
import numpy as np

image_paths = ["cat1.jpg", "cat2.jpg", "cat3.jpg"]
output_csv = "new_patterns.csv"

all_images = []

for image_path in image_paths:
    image = Image.open(image_path)
    image = image.resize((100, 100))
    image_gray = image.convert("L")
    image_binary = image_gray.point(lambda p: 1 if p < 128 else -1)
    binary_array = np.array(image_binary).flatten()
    all_images.append(binary_array)

np.savetxt(output_csv, all_images, delimiter=",", fmt="%d")

data = np.loadtxt(output_csv, delimiter=",", dtype=int)
data[data == 0] = -1
np.savetxt(output_csv, data, delimiter=",", fmt="%d")
