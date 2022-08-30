import os
from shutil import rmtree

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import warnings
import CustomErrors
warnings.filterwarnings("ignore", category=UserWarning) 

# Use GPU whenever available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0) # Swap matrix to 224 * 224 * 1
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) ## Denormalize
  image = image.clip(0, 1) # Set range between 0 and 1
  return image

class Model:
    def __init__(self):
        # Transformation to training set
        self.transform_train = transforms.Compose([ 
                            transforms.Resize((224,224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                            transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        
        # Create a transformation chain to convert the image to Tensor format
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

    def load_dataset(self, folder_path, batch_size):
        try:
            # Check if both folders exist first
            training_path = f"{folder_path}/train"
            val_path = f"{folder_path}/val"

            training_classes = [f.name for f in os.scandir(training_path) if f.is_dir()]
            val_classes = [f.name for f in os.scandir(val_path) if f.is_dir()]

            if len(training_classes) != len(val_classes) or len(set(training_classes) ^ set(val_classes)) != 0:
                raise CustomErrors.ClassMismatchError

            # For training
            self.training_dataset = datasets.ImageFolder(training_path, transform=self.transform_train)
            # For testing
            self.validation_dataset = datasets.ImageFolder(val_path, transform=self.transform)

            self.training_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

            # Classes
            self.classes = tuple(training_classes)

            print(f"+ Imported {len(self.training_dataset)} training images")
            print(f"+ Imported {len(self.validation_dataset)} validation images")
            print(f"+ Classes: {self.classes}")
        
        except FileNotFoundError:
            print(f"- Either train or val folder does not exist")
        
        except CustomErrors.ClassMismatchError:
            print(f"- Mismatched classes in training and validation fodlers")
    
    def show_training_samples(self):
        # TODO show a few training samples
        pass

    
    def save_model(self, path):
        # TODO Refine to save epochs and parameters?
        if self.model:
            torch.save(self.model, path)
        else:
            print(f"- Model not yet initialized")

    def load_model(self, model_type = None, custom_model = False):
        # e.g. load models/supervised_benchmark.pt

        if custom_model == True:
            try:
                self.model = torch.load(model_type)
                self.model.eval()
            except FileNotFoundError:
                print(f"- load_model() -> File not found")
        else:
            try:
                if model_type == "vgg19":
                    self.model = models.vgg19(pretrained = True)
            except:
                print(f"- load_model() -> Failed to load model")

        print(f"+ Model loaded")


    def transfer_learning_init(self):
        if hasattr(self, 'model') == False:
            print(f"- Model not yet initialized.")
        elif hasattr(self, 'classes') == False:
            print(f"- Classes not yet initialized.")
        else:
            # Freeze Feature Extractor
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Add a layer to the end to swap output node to the number of classes in our dataset
            n_inputs = self.model.classifier[6].in_features
            last_layer = nn.Linear(n_inputs, len(self.classes))
            self.model.classifier[6] = last_layer
            self.model.to(device)
            print(f"+ {self.model.classifier[6].out_features}")
            

    def train_model(self, learning_rate, epochs):
        if hasattr(self, 'training_loader') == False or hasattr(self, 'validation_loader') == False:
            print(f"- train_model() -> Dataset not yet loaded")
            return
        
        if hasattr(self, 'model') == False:
            print(f"- train_model() -> Model not yet loaded")
            return

        if len([f.name for f in os.scandir('temp_model_step') if f.is_file()]) != 0:
            # Reset cache
            rmtree('temp_model_step')
            os.mkdir('temp_model_step')
            
        criterion = nn.CrossEntropyLoss()
        # If the learning rate is too large, the accuracy might step too much and jumped up and down.
        # When this happens, decrease the learning rate to get a smoother curve
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
    
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []

        for e in range(epochs):
            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0
            
            for inputs, labels in tqdm(self.training_loader, unit="batch"):
                # Pass to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
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
                    for val_inputs, val_labels in self.validation_loader:
                        # Send to GPU
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = self.model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)
                        
                        _, val_preds = torch.max(val_outputs, 1) # Top class predictions for each image
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(val_preds == val_labels.data)
                    
                    # Get average accuracy for the epoch
                    epoch_loss = running_loss/len(self.training_loader.dataset)
                    epoch_acc = running_corrects.float()/ len(self.training_loader.dataset)
                    self.running_loss_history.append(epoch_loss)
                    self.running_corrects_history.append(epoch_acc.cpu())
                    
                    val_epoch_loss = val_running_loss/len(self.validation_loader.dataset)
                    val_epoch_acc = val_running_corrects.float()/ len(self.validation_loader.dataset)
                    self.val_running_loss_history.append(val_epoch_loss)
                    self.val_running_corrects_history.append(val_epoch_acc.cpu())
                    print('epoch :', (e+1))
                    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
                    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

            # Save epoch
            temp_save = f"temp_model_step/epoch{e+1}.pt"
            torch.save(self.model, temp_save)

    def plot_loss(self, path = 'results/loss.png'):
        if hasattr(self, 'running_loss_history') == False or hasattr(self, 'val_running_loss_history'):
            print(f"- plot_loss() -> No data available to plot")
            return

        loss_fig = plt.figure()
        loss_ax = loss_fig.add_subplot(1, 1, 1)

        loss_ax.plot(self.running_loss_history, label='training loss')
        loss_ax.plot(self.val_running_loss_history, label='validation loss')
        loss_ax.legend()

        loss_fig.savefig(path, dpi=loss_fig.dpi)

    def plot_accuracy(self, path = 'results/acc.png'):
        if hasattr(self, 'running_corrects_history') == False or hasattr(self, 'val_running_corrects_history'):
            print(f"- plot_accuracy() -> No data available to plot")
            return

        acc_fig = plt.figure()
        acc_ax = acc_fig.add_subplot(1, 1, 1)

        acc_ax.plot(self.running_corrects_history, label='training accuracy')
        acc_ax.plot(self.val_running_corrects_history, label='validation accuracy')
        acc_ax.legend()

        acc_fig.savefig(path, dpi=acc_fig.dpi)

    def test_model(self, path = 'results/test.png'):
        if hasattr(self, 'model') == False:
            print(f"- Model not yet loaded")
            return

        elif hasattr(self, 'validation_dataset') == False:
            print(f"- Dataset not yet loaded")
            return
        
        test_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=20, shuffle=True)
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        output = self.model(images)
        _, preds = torch.max(output, 1)

        end_fig = plt.figure(figsize=(25, 4))

        for idx in np.arange(20):
            ax = end_fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            plt.imshow(im_convert(images[idx]))
            if preds[idx]==labels[idx]:
                ax.set_title("{}".format(str(self.classes[preds[idx].item()])), color="green")
            else:
                ax.set_title("{} ({})".format(str(self.classes[preds[idx].item()]), str(self.classes[labels[idx].item()])), color="red")
        end_fig.savefig(path, dpi=end_fig.dpi)
        