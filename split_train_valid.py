import os
import shutil
import random
random.seed(0)
label = os.listdir('./TINYIMAGENET/train/')

source = './TINYIMAGENET/valid/'
for class_id in label:
    imgs = os.listdir(f'./TINYIMAGENET/train/{class_id}/images')
    selected_items = random.sample(imgs, 100)

    if not os.path.exists(os.path.join(source, class_id, 'images')):
        os.makedirs(os.path.join(source, class_id, 'images'))

    for img in selected_items:
        shutil.move(os.path.join(f'./TINYIMAGENET/train/{class_id}/images', img), os.path.join(source, class_id, 'images', img))
    

