# deepgaldx - Or: How to generate fake Galdxs?

## Background: What is Galdx? 
Me and a few friends of mine organized a "Science-Slam"-like event, in which we can present some stuff that we find interesting, and which allow the audience to understand and hopefully share our fascination in a specific topic. Its a friends-only event---nothing fancy and official---and we called it "Deus Academicus" (since we are bad at latin).
I proposed a few topics that didn't really convince my friends. "Can't you just generate some deep fakes of Galdx (another friend of mine; I censored the name due to privacy reasons)?" asked someone since I am the Deep--Learning guy in our clique. 
Challange Accepted! 
I decided to apply a deep learning architecture (Cycle-GANs) to transfer the style of realstic images of faces to anime faces. Unlike standard deep generative models, we try to retain as many characteristics of the realistic face image in our generated anime face as possible. 

The presentation slides are in `presentation/deep_galdx.pdf` and cover a high-level overview about i) artificial neural networks, ii) generative adversarial networks (GANs), iii) Cylce-GANs. 
Note: While the presentation itself does not require any knowledge about Deep--Learning, the slides itself are mostly not self-explanatory. Additionally, I tried to avoid as many formulas and technicalities to make the talk as accessible as possible.

# How to use the Software? 
This repository uses PyTorch to train, evaluate, and sample from the Cycle-GAN, where a majority of the code is taken from the an implementation by [Lornatang](https://github.com/Lornatang/CycleGAN-PyTorch). The original paper can be found [here](https://arxiv.org/abs/1703.10593).  

## Step 1: Downloading and Preprocessing the data 
We require 2 datasets, each representing images in a certain style. 

To obtain images of real faces, I decided to use the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Download and extract the Align&Cropped Images (Img/img_align_celeba.zip in the corresponding Google Drive) into the `data/` directory.

A dataset containing anime faces can be found [here](https://drive.google.com/file/d/1HG7YnakUkjaxtNMclbl2t5sJwGLcHYsI/view)
Again, download and extract the data into `data/`.

The software taken from Cycle-GAN requires the data to be organized in directories: 
| Directory | Contains... |
| :------- | ----: | 
|data/train/A | Training data in style A|
|data/train/B | Training data in style B|
|data/test/A | Test data in style A|
|data/test/B | Test data in style B |

```
cd data/celeb2manga
python3 process_anime.py
python3 process_celeba.py
```

to reorganize the data to the required structure. 

## Step 2: Train the Cylce-GAN
`cd ../cyclegan/`
to direct to the folder containing the code for training. Again, I want to emphasize that most of the code in this directory is taken from the implementation by Lornatang.
To train the code, run 
```python3 cyclegan.py ```
with a range of optional arguments: 
```
python3 cyclegan.py -h
usage: cyclegan.py [-h] [--epoch EPOCH] [--n_epochs N_EPOCHS]
                   [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2]
                   [--decay_epoch DECAY_EPOCH] [--n_cpu N_CPU]
                   [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH]
                   [--channels CHANNELS] [--sample_interval SAMPLE_INTERVAL]
                   [--checkpoint_interval CHECKPOINT_INTERVAL]
                   [--n_residual_blocks N_RESIDUAL_BLOCKS]
                   [--lambda_cyc LAMBDA_CYC] [--lambda_id LAMBDA_ID]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         epoch to start training from
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --decay_epoch DECAY_EPOCH
                        epoch from which to start lr decay
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --img_height IMG_HEIGHT
                        size of image height
  --img_width IMG_WIDTH
                        size of image width
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        interval between saving generator outputs
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between saving model checkpoints
  --n_residual_blocks N_RESIDUAL_BLOCKS
                        number of residual blocks in generator
  --lambda_cyc LAMBDA_CYC
                        cycle loss weight
  --lambda_id LAMBDA_ID
                        identity loss weight
```

The model shown in the presentation resulted from training for 10 epochs and using all default values:
```
python3 cyclegan.py --epoch 10
```

To validate the training process, the code traces generated images in `cyclegan/images/celeb2manga/`. After each epoch, we store the model in `cyclegan/saved_models/celeb2manga/`.

## Step 3: Sampling Galdxs from a trained model
Real images of Galdx (or whose persons face you want to transform into an anime face) need to be stored in `cyclegan/images/galdx/`. 
Finally, to transform all images in `cyclegan/images/galdx/` using the trained model after $j$ epochs, we run 
```
python3 sample_galdx.py --epoch $j
```
Run 
```
./generate_galdxs.sh 
```
to transform images using the trained models after $j \in \{1,...,10\}$ epochs. 
All transformed faces are stored in `cyclegan/images/galdx2manga/`. 

# Discussion: 

