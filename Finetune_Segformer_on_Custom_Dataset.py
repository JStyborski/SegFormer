import requests, zipfile, io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Gathering Datasets
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Custom Dataset class
from Dataset import SemanticSegmentationDataset

# Model
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# HuggingFace Metrics
import evaluate

# Custom palette
from Palette import ade_palette

# Torch stuff
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

# Original File: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb
# In this notebook, we are going to fine-tune `SegFormerForSemanticSegmentation` on a custom **semantic segmentation**
# dataset. In semantic segmentation, the goal for the model is to label each pixel of an image with one of a list of
# predefined classes. We load the encoder of the model with weights pre-trained on ImageNet-1k, and fine-tune it
# together with the decoder head, which starts with randomly initialized weights.

# Download a small subset of the ADE20k dataset, an important benchmark for semantic segmentation
# It contains 150 labels

# I've made a small subset just for demonstration purposes (namely the 10 first training and 10 first validation images
# + segmentation maps).
# The goal for the model is to overfit this tiny dataset (because that makes sure that it'll work on a larger scale).

#################
# Download Data #
#################

print('Downloading Data')

def download_data():
    url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

download_data()

# Can also load the entire dataset (a few GBs)
load_entire_dataset = False
if load_entire_dataset:
    dataset = load_dataset("scene_parse_150")

##################################
# Define Dataset and Dataloaders #
##################################

print('Defining Datasets and Dataloaders')

# Initialize the training + validation datasets. Important: we initialize the feature extractor with
# `reduce_labels=True`, as the classes in ADE20k go from 0 to 150, with 0 meaning "background". However, we want the
# labels to go from 0 to 149, and only train the model to recognize the 150 classes (which don't include "background").
# Hence, we'll reduce all labels by 1 and replace 0 by 255, which is the `ignore_index` of SegFormer's loss function.

root_dir = 'ADE20k_toy_dataset'
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

# Create datasets - Each dataset is a dictionary with keys 'pixel_values' and 'labels'
# dict['pixel_values'] is a torch tensor of 3x512x512 decimal values centered around 0
# dict['labels'] is a torch tensor 512x512 of ~20 different integers between 0 and 255 (representing classes)
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)

print('  Number of training examples:', len(train_dataset))
print('  Number of validation examples:', len(valid_dataset))

# Create dataloaders - each will provide PyTorch tensor batches which contains 'pixel_values' and 'labels' keys
# The batch['pixel_values'] is a torch tensor of batchsizex3x512x512
# The batch['labels'] is a torch tensor of batchsizex512x512
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)

####################
# Model Definition #
####################

print('Loading Model')

# Load the model and equip the encoder with weights pre-trained on ImageNet-1k (we take the smallest variant,
# `nvidia/mit-b0` here, but you can take a bigger one https://huggingface.co/models?other=segformer

# Set the `id2label` and `label2id` mappings, used when performing inference, from JSON on the hub
# Both are dictionaries with 150 entries - id2label has entries 0: 'wall', 1: 'building' ... and label2id is opposite
repo_id = 'huggingface/label-files'
filename = 'ade20k-id2label.json'
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# Instantiate model with label dictionaries
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=150, id2label=id2label, label2id=label2id)

######################
# Finetune the Model #
######################

print('Finetuning Model')

# Track metrics during training. For semantic segmentation, typical metrics include the mean intersection-over-union
# (mIoU) and pixel-wise accuracy. These are available in the Datasets library

metric = evaluate.load('mean_iou')

# Fine-tune the model in native PyTorch, using the AdamW optimizer
# Use the same learning rate as the one reported in the [paper](https://arxiv.org/abs/2105.15203)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model in training mode
model.train()

# Loop over the training data
for epoch in range(10):
    # Loop through batches of data
    for batch in tqdm(train_dataloader):

        # Get inputs
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward propagation
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the output logits
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
          
            # Note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    # Compute metrics from the added batches - Note we've already reduced the labels before
    metrics = metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)

    print('  {} | Loss: {} | Mean IoU: {} | Mean Acc: {}'.format(epoch, loss.item(), metrics['mean_iou'],
                                                               metrics['mean_accuracy']))

###################
# Model Inference #
###################

print('Predicted Segmentation')

# Import image to test (The ADE_train_00000001 pic is 512Hx683W with 3 color channels)
origImage = Image.open('ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg')
#image.show()

# Prepare the image for the model
# encoding is a batch dictionary with key 'pixel_values' and values as a 1x3x512x512 PyTorch tensor
encoding = feature_extractor(origImage, return_tensors="pt")
pixel_values = encoding['pixel_values'].to(device)

# Forward propagation through model
# logits are a PyTorch tensor with shape 1x150x128x128 (note 1/4H and 1/4W)
model.eval()
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()

# Rescale logits to original image size - note that size is reversed to HxW
upsampled_logits = nn.functional.interpolate(logits, size=origImage.size[::-1], mode='bilinear', align_corners=False)

# Apply argmax on the class dimension to get most likely class for each pixel
predSegClass = upsampled_logits.argmax(dim=1)[0]

# Create a numpy array HxWx3 to contain the segmentation map colors for each selected class
colorPredSeg = np.zeros((predSegClass.shape[0], predSegClass.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(np.array(ade_palette())):
    colorPredSeg[predSegClass == label, :] = color

# Convert to BGR
colorPredSeg = colorPredSeg[..., ::-1]

# Show image + mask
predImg = np.array(origImage) * 0.5 + colorPredSeg * 0.5
predImg = predImg.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(predImg)
plt.show()

###########################
# Ground Truth Comparison #
###########################

print('Ground Truth Segmentation')

# Import ground truth segmentation image
segImage = Image.open('ADE20k_toy_dataset/annotations/training/ADE_train_00000001.png')

# Modify the class indices - 0 (background class) becomes 255 (ignore_index) and all of the class are shifted down
segImage = np.array(segImage)
segImage[segImage == 0] = 255 # background class is replaced by ignore_index
segImage = segImage - 1 # other classes are reduced by one
segImage[segImage == 254] = 255

classes_map = np.unique(segImage).tolist()
unique_classes = [model.config.id2label[idx] if idx!=255 else None for idx in classes_map]
print('  Classes in this image:', unique_classes)

# Create coloured map
colorTrueSeg = np.zeros((segImage.shape[0], segImage.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(np.array(ade_palette())):
    colorTrueSeg[segImage == label, :] = color

# Convert to BGR
colorTrueSeg = colorTrueSeg[..., ::-1]

# Show image + mask
img = np.array(origImage) * 0.5 + colorTrueSeg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

# Some random querying
#predSegClass.unique()
#model.config.id2label[37]
#np.unique(segImage)
#predSegClass
#segImage

###################
# Compute Metrics #
###################

print('Compute Prediction Metrics')

# metric expects a list of numpy arrays for both predictions and references
# metrics is a dict with keys 'mean_iou', 'mean_accuracy', 'overall_accuracy', 'per_category_iou', and 'per_category_accuracy'
metrics = metric.compute(predictions=[predSegClass.numpy()], references=[segImage], num_labels=150, ignore_index=255)

# Print overall metrics
for key in list(metrics.keys())[:3]:
  print('  ' + key, metrics[key])

# Print per category metrics as Pandas DataFrame
metric_table = dict()
for idx, label in id2label.items():
    metric_table[label] = [metrics['per_category_iou'][idx], metrics['per_category_accuracy'][idx]]

print('  Per-Category Metrics:')
perCatMetrics = pd.DataFrame.from_dict(metric_table, orient="index", columns=["IoU", "accuracy"])
print(perCatMetrics.head())
