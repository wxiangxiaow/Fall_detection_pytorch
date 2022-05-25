# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import torch
import seaborn as sns
import matplotlib as plt
plt.use('agg')
import cv2
import os

class_to_idx = {'No fall / Not squat': 0, 'Squat': 1, 'Fall':2}
cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}

def process_image(image):
    #transfer dcm to Image
    file_name = cv2.imread(image)
    img = Image.fromarray(file_name).convert('RGB')
    #Scales, crops, and normalizes a PIL image for a PyTorch model,returns an Numpy array
    ##########Scales 
    if img.size[0] > img.size[1]:
        img.thumbnail((1000000, 256))
    else:
        img.thumbnail((256 ,1000000))
    #######Crops: to crop the image we have to specifiy the left,Right,button and the top pixels because the crop function take a rectongle ot pixels
    Left = (img.width - 224) / 2
    Right = Left + 224
    Top = (img.height - 244) / 2
    Buttom = Top + 224
    img = img.crop((Left, Top, Right, Buttom))

    #normalization (divide the image by 255 so the value of the channels will be between 0 and 1 and substract the mean and divide the result by the standtared deviation)
    img = ((np.array(img) / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.pyplot.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    #image=np.transpose(image)
    ax.imshow(image)
    #return image

def predict(image_path, model, device, topk=3):
    #Predict the class (or classes) of an image using a trained deep learning model.
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    model_input = model_input.to(device)
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    
    # Convert indices to classes
    top_flowers = [cat_to_name[lab] for lab in top_labs]

    return top_probs, top_flowers

    
def plot(image_path,model,device, image_name, top_k=3):
    proba, flowers = predict(image_path, model, device, top_k)
    plt.pyplot.figure(figsize=(6,10))
    ax = plt.pyplot.subplot(2,1,1)
    
    title = image_name
    imshow(process_image(image_path), ax, title=title)
    
    plt.pyplot.subplot(2,1,2)
    sns.barplot(x=proba, y=flowers, color=sns.color_palette()[0])
    plt.pyplot.savefig('./pictures/'+image_name+'-predict_img.png')


def judge(image_dir,model,device,top_k=3):
    #store img path
    imgs = []
    #store point
    none_point = 0
    squat_point = 0
    fall_point = 0
    for i in os.listdir(image_dir):
        imgs.append(i)
    imgs.sort()
    for i in imgs:
        image_path = image_dir + i
        proba, flowers = predict(image_path, model, device, top_k)
        #print(flowers[0])
        if flowers[0]=='Fall':
            fall_point += 1
        elif flowers[0]=='Squat':
            squat_point += 1
        else:
            none_point += 1
    point_list = [none_point,squat_point,fall_point]
    max_point = max(point_list)
    max_posi = point_list.index(max_point)
    name = ['none','squat','fall']
    print('video recogntion result:'+name[max_posi]+':'+str(max_point))
    for i in range(3):
        print(point_list[i])
    




if __name__ == '__main__':
    test_dir = './test_image/'
    print('now test image folder='+test_dir)
    checkpoint = torch.load('./checkpoint/BEST_checkpoint.pth.tar')
    model = checkpoint['model']
    print('load model...')
    device = torch.device('cuda:1')
    print('device num='+str(device))
    for i in os.listdir(test_dir):
        img_path = test_dir+i
        img_name = i.split('.')[0]
        #print('now image='+img_path)
        plot(img_path,model,device,img_name,3)


