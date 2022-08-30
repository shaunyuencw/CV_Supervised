import os
from shutil import copyfile 
from shutil import rmtree
from tqdm import tqdm
import CustomErrors

folder_path = 'minibatch_data'

class Dataset_Generator():
    def __init__(self, main_path):
        try:
            # Check if both folders exist first
            training_path = f"{main_path}/train"
            val_path = f"{main_path}/val"

            self.training_classes = [f.path for f in os.scandir(training_path) if f.is_dir()]
            self.val_classes = [f.path for f in os.scandir(val_path) if f.is_dir()]

            t_classes = [f.name for f in os.scandir(training_path) if f.is_dir()]
            v_classes = [f.name for f in os.scandir(val_path) if f.is_dir()]

            if len(t_classes) != len(v_classes) or len(set(t_classes) ^ set(v_classes)) != 0:
                raise CustomErrors.ClassMismatchError

            
        except CustomErrors.ClassMismatchError:
            print(f"- Number of classes in train folder does not equals val folder")

    def reset_minibatch(self, minibatch_path):
        rmtree(f"{minibatch_path}/train")
        rmtree(f"{minibatch_path}/val")
        os.mkdir(f"{minibatch_path}/train")
        os.mkdir(f"{minibatch_path}/val")
        print(f"+ minibatch folder resetted")

    def generate_minibatch(self, size, minibatch_path, scaled=True):
        # If folders are not empty, reset
        if len([f.path for f in os.scandir('minibatch_data/train') if f.is_dir()]) != 0 or len([f.path for f in os.scandir('minibatch_data/val') if f.is_dir()]) != 0:
            self.reset_minibatch('minibatch_data')

        train_images = 0
        val_images = 0

        # Process training images
        for class_path in tqdm(self.training_classes, unit="Train Classes"):
            images = [f.path for f in os.scandir(class_path) if f.is_file()]
            mini_class = class_path.replace('data', minibatch_path)

            os.mkdir(mini_class)
            counter = 0

            if scaled:
                minibatch_size = len(images) * size
            else:
                minibatch_size = size

            for image in images:
                if counter >= minibatch_size:
                    break

                # Copy to minibatch folder
                target = image.replace('data', 'minibatch_data')
                #print(f"+ {image}")
                copyfile(image, target)

                # print(f"Image {counter}: {image}")
                # print(target)
                counter += 1
                train_images += 1

        # Process val images
        for class_path in tqdm(self.val_classes, unit="Val Classes"):
            images = [f.path for f in os.scandir(class_path) if f.is_file()]
            mini_class = class_path.replace('data', minibatch_path)

            os.mkdir(mini_class)
            counter = 0

            if scaled:
                minibatch_size = len(images) * size
            else:
                minibatch_size = size

            for image in images:
                if counter >= minibatch_size:
                    break

                # Copy to minibatch folder
                target = image.replace('data', 'minibatch_data')
                #print(f"+ {image}")
                copyfile(image, target)

                # print(f"Image {counter}: {image}")
                # print(target)
                counter += 1
                val_images += 1


        print(f"+ Minibatch_data has {train_images} Training images and {val_images} Validaiton images")