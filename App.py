import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0) # Swap matrix to 224 * 224 * 1
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) ## Denormalize
  image = image.clip(0, 1) # Set range between 0 and 1
  return image

# Use GPU whenever available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformation to training set
transform_train = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

# Create a transformation chain to convert the image to Tensor format
transform = transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

# For training
training_dataset = datasets.ImageFolder('data/train', transform=transform_train)
# For testing
validation_dataset = datasets.ImageFolder('data/val', transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=250, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=250, shuffle=False)

print(f"+ Imported {len(training_dataset)} training images")
print(f"+ Imported {len(validation_dataset)} validation images")

# Classes in dataset
classes = tuple([f.name for f in os.scandir('data/train') if f.is_dir()])
print(f"+ Classes: {classes}")

#print(f"+ Displaying some samples")
# dataiter = iter(training_loader)
# images, labels = dataiter.next()
# fig = plt.figure(figsize=(25, 4))

# # Display some images from training dataset
# for idx in np.arange(20):
#   ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
#   plt.imshow(im_convert(images[idx]))
#   ax.set_title(classes[labels[idx].item()])

#fig.savefig('temp.png', dpi=fig.dpi)

# Load VGG19 Model
model = models.vgg19(pretrained = True)
#print(model)

# Freeze Feature Extractor
for param in model.features.parameters():
  param.requires_grad = False

import torch.nn as nn

# Add a layer to the end to swap output node to the number of classes in our dataset
n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer
model.to(device)
print(model.classifier[6].out_features)

criterion = nn.CrossEntropyLoss()
# If the learning rate is too large, the accuracy might step too much and jumped up and down.
# When this happens, decrease the learning rate to get a smoother curve
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005)

epochs = 25
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in tqdm(training_loader, unit="batch"):
    # Pass to GPU
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad() # Reset
    loss.backward() # Derivative
    optimizer.step() # Update parameters
    
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)

  else:
    with torch.no_grad(): # Temp set all require_grad flags to False to save memory
      # Compute validation
      for val_inputs, val_labels in validation_loader:
        # Send to GPU
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        
        _, val_preds = torch.max(val_outputs, 1) # Top class predictions for each image
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
    
    # Get average accuracy for the epoch
    epoch_loss = running_loss/len(training_loader.dataset)
    epoch_acc = running_corrects.float()/ len(training_loader.dataset)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc.cpu())
    
    val_epoch_loss = val_running_loss/len(validation_loader.dataset)
    val_epoch_acc = val_running_corrects.float()/ len(validation_loader.dataset)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc.cpu())
    print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

loss_fig = plt.figure()
loss_ax = loss_fig.add_subplot(1, 1, 1)

loss_ax.plot(running_loss_history, label='training loss')
loss_ax.plot(val_running_loss_history, label='validation loss')
loss_ax.legend()

loss_fig.savefig('results/loss.png', dpi=loss_fig.dpi)

acc_fig = plt.figure()
acc_ax = acc_fig.add_subplot(1, 1, 1)

acc_ax.plot(running_corrects_history, label='training accuracy')
acc_ax.plot(val_running_corrects_history, label='validation accuracy')
acc_ax.legend()

acc_fig.savefig('results/acc.png', dpi=loss_fig.dpi)

dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

end_fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = end_fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  if preds[idx]==labels[idx]:
       ax.set_title("{}".format(str(classes[preds[idx].item()])), color="green")
  else:
       ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color="red")
end_fig.savefig('results/test.png', dpi=loss_fig.dpi)

PATH = "models/supervised_benchmark.pt"
torch.save(model, PATH)

# Load Model
# model = torch.load(PATH)
# model.eval()
