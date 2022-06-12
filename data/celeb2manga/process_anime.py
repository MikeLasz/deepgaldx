import glob
import random
import os
from PIL import Image

root = "animeface-character-dataset"
manga_data = sorted(glob.glob(os.path.join(root, "data") + "/*.*"))

train_set_id = random.sample(range(len(manga_data)), int(0.9*len(manga_data)))
test_set_id = list(set(range(len(manga_data))) - set(train_set_id))
for id in train_set_id:
	image = Image.open(manga_data[id])
	image.save(f"train/A/{id}.png")
	
for id in test_set_id:
	image = Image.open(manga_data[id])
	image.save(f"test/A/{id}.png")
	
	
	

