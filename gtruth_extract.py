import json
import cv2
import os
import pandas as pd
from pose_est import angle_det

folder = 'imgs/'

images_list = os.listdir(folder)
angle_list = []

for image in images_list:
    img = cv2.imread(folder+image)

    img, joint_dic = angle_det(img)

    cv2.imwrite('joints_res/'+image, img)

    angle_list.append(joint_dic)

with open('joints_res/angles_gtruth.txt', 'w') as fout:
    json.dump(angle_list, fout)

json_file = pd.read_json('joints_res/angles_gtruth.txt')

json_file.to_csv('joints_res/angles_gtruth.csv')