import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import skimage
from matplotlib.colors import hsv_to_rgb
import os 

FRAME_PAIR_DELTA_MS = 300
IMG_HEIGHT = 66
IMG_WIDTH = 200
NUM_TO_PROCESS = 1000

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    h,w = image_current.shape
    hsv = np.zeros((h,w,3)).astype(np.uint8)
    hsv[...,1] = 255
    
    flow = cv2.calcOpticalFlowFarneback(image_current,image_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return rgb

info_paths = glob.glob('data/info/*')[:NUM_TO_PROCESS]
total_to_process = len(info_paths)
i = 1
os.makedirs('data/flow/', exist_ok=True)
with open('data/info.csv', 'w+') as f_info:
    f_info.write('path,latitude,longitude,speed\n')
    for info_path in info_paths:
        print('{}/{}'.format(i, total_to_process))
        vid_path = info_path.replace('info','videos').replace('.json','.mov')
        with open(info_path) as f:
            info = json.load(f)
        try:
            info_df = pd.DataFrame(info['locations'])
            info_df['timestamp'] -= min(info_df['timestamp'])
            vid = cv2.VideoCapture(vid_path)
        except:
            continue
    #     print('Reading {}...'.format(vid_path))
        for index, row in info_df.iterrows():
            if index == 0:
                continue
            timestamp = row['timestamp']
            vid.set(cv2.CAP_PROP_POS_MSEC, timestamp - FRAME_PAIR_DELTA_MS)
            success, frame = vid.read()
            vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            next_success, next_frame = vid.read()
            if success and next_success:
                # generate flow image
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
                next_frame = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
                next_frame = cv2.resize(next_frame, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
#                 frame_filename = 'data/frames/{}_{}.jpg'.format(vid_path.split('/')[-1][:-4],timestamp)
#                 next_frame_filename = 'data/frames/{}_{}.jpg'.format(vid_path.split('/')[-1][:-4],timestamp+FRAME_PAIR_DELTA_MS)
                
                flow = opticalFlowDense(frame, next_frame)
                flow_filename = 'data/flow/{}_{}.jpg'.format(vid_path.split('/')[-1][:-4],timestamp)
#                 cv2.imwrite(frame_filename, frame)
#                 cv2.imwrite(next_frame_filename, next_frame)
                cv2.imwrite(flow_filename, flow)
                
    #             X.append(flow)
                # compute gps delta 
                lat_delta = (info_df.loc[index]['latitude'] - info_df.loc[index-1]['latitude'])*100000
                lon_delta = (info_df.loc[index]['longitude'] - info_df.loc[index-1]['longitude'])*100000
                speed = info_df.loc[index]['speed']
                f_info.write('{},{},{},{}\n'.format(flow_filename,lat_delta, lon_delta, speed))
        i += 1