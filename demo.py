from video2img import path2name
from predict import judge
import torch
import os

path1 = './test_video/videos/'

if __name__ == '__main__':
	img_list = []
	for i in os.listdir(path1):
		video_path = path1+i
		img_path = path2name(video_path)
		img_list.append(img_path)
	checkpoint = torch.load('./checkpoint/checkpoint.pth.tar')
	model = checkpoint['model']
	print('load model...')
	device = torch.device('cuda:1')
	print('device:'+str(device))
	for i in img_list:
		judge(i,model,device,3)
