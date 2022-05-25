import cv2
import os

output_dir = './test_video/frames/'

#save frames	
def read_the_video(video_name, video_path):
	vc = cv2.VideoCapture(video_path)
	print("now reading video path:"+video_path)
	if vc.isOpened():
		ret, frame = vc.read()
		print("save 1 frame from 3 frames")
		c = 1
		while ret:
			#print('save frames:'+str(c))
			if c%3 == 1:
				cv2.imwrite(output_dir+video_name+'/'+video_name+'-'+str(c)+'.png',frame)
			c = c+1
			ret, frame = vc.read()
	vc.release()


#turn path to name
def path2name(video_path):
	video_name = video_path.split('/')[-1]
	comment = video_name.split('.')
	if len(comment)==2:
		video_name = comment[0]
		output_path = output_dir + video_name + '/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		read_the_video(video_name,video_path)
	else:
		print('error:invalid video path.')
	return output_path
