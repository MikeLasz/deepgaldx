import glob
import random
import os
from PIL import Image

root = "celeba/img_align_celeba"
celeb_data = sorted(glob.glob(root + "/*.*"))

train_set_id = random.sample(range(len(celeb_data)), int(0.9*len(celeb_data)))
test_set_id = list(set(range(len(celeb_data))) - set(train_set_id))
for id in train_set_id:
	image = Image.open(celeb_data[id])
	image.save(f"train/B/{id}.png")
	
for id in test_set_id:
	image = Image.open(celeb_data[id])
	image.save(f"test/B/{id}.png")
	
