import os
from shutil import copyfile 

classes = [f.path for f in os.scandir('data/train') if f.is_dir()]

for class_path in classes:
    images = [f.path for f in os.scandir(class_path) if f.is_file()]
    mini_class = class_path.replace('data', 'minibatch_data')
    os.mkdir(mini_class)
    counter = 0
    scaled_ratio = 0.5
    mini_batch_size = len(images) * scaled_ratio
    for image in images:
        if counter >= mini_batch_size:
            break

        # Copy to minibatch folder
        target = image.replace('data', 'minibatch_data')
        #print(f"+ {image}")
        copyfile(image, target)

        # print(f"Image {counter}: {image}")
        # print(target)
        counter += 1