# -*- coding: utf-8 -*-
'''
    File name: DataGenerator.py
    Author: Rasool Sabzi
    Date created: 9/5/2018
    Date last modified: 9/7/2018
    Python Version: 3.6
'''

import numpy as np
import trimesh
import argparse
import os

def Generate2D(input_path,output_path,degree=40,resolution=[100, 100],color=[128,128,128,255]):
    for Class in os.listdir(input_path):
        for Folder in os.listdir(input_path+Class):
            for vox in os.listdir(input_path+Class+'\\'+Folder):
                if vox.endswith('.off'):
                    mesh = trimesh.load(input_path+Class+'\\'+Folder+vox)
                    mesh.visual.face_colors =  np.array(color)
                    scene = mesh.scene()
                    Mrotate = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0],scene.centroid)
                    mesh.apply_transform(Mrotate)
                    rotate = trimesh.transformations.rotation_matrix(np.radians(40), [0, 1, 0],scene.centroid)
                    for i in range(360//degree):
                        mesh.apply_transform(rotate)
                        file_name = vox.split('.')[0]+'_'+str((i+1)*degree)+ '.png'
                        png = scene.save_image(resolution=resolution, visible=True)
                        if (Folder=='train'):
                            savePath=output_path+'\\train\\'+file_name
                        else:
                            savePath=output_path+'\\test\\'+file_name
                        with open(savePath, 'wb') as f:
                            f.write(png)
                            f.close()
                        print(vox)