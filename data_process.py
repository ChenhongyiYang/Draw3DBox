from PIL import Image
import numpy as np
import glob
import os
import random


H, W = 224, 224
DIRECTIONS = 2


def parse_annotation(label_dir,image_dir):
    all_objs = []
    rs_mean = [0.,0.]
    theta1_mean = [0., 0.]
    theta2_mean = [0., 0.]
    dir_0, dir_1 = 0., 0.
    for label_file in glob.glob(label_dir+'/*'):
        img_name = label_file.split('/')[-1].split('.')[0]
        for line in open(label_file).readlines():
            line = line.strip().split(' ')

            obj = {'name': line[0],
                   'image': image_dir+'/'+img_name+'.png',
                   'xmin': int(float(line[1])),
                   'ymin': int(float(line[2])),
                   'xmax': int(float(line[3])),
                   'ymax': int(float(line[4])),
                   'rs': float(line[5]),
                   'theta1': float(line[6]),
                   'theta2': float(line[7]),
                   'direction': int(line[8])
                   }
            if obj['rs'] < 1.0 and obj['rs'] > 0.0:
                all_objs.append(obj)
            if obj['direction'] == 0:
                rs_mean[0] = (rs_mean[0] * dir_0 + obj['rs']) / (dir_0+1.)
                theta1_mean[0] = (theta1_mean[0] * dir_0 + obj['theta1']) / (dir_0 + 1.)
                theta2_mean[0] = (theta2_mean[0] * dir_0 + obj['theta2']) / (dir_0 + 1.)
                dir_0 += 1.0
            else:
                rs_mean[1] = (rs_mean[1] * dir_1 + obj['rs']) / (dir_1 + 1.)
                theta1_mean[1] = (theta1_mean[1] * dir_1 + obj['theta1']) / (dir_1 + 1.)
                theta2_mean[1] = (theta2_mean[1] * dir_1 + obj['theta2']) / (dir_1 + 1.)
                dir_1 += 1.0

    fm = open('data_mean.txt','w')
    fm.write('%f %f\n'%(rs_mean[0], rs_mean[1]))
    fm.write('%f %f\n'%(theta1_mean[0], theta1_mean[1]))
    fm.write('%f %f'%(theta2_mean[0], theta2_mean[1]))
    fm.close()


    for obj in all_objs:
        t1_train = np.zeros([DIRECTIONS,2])
        t2_train = np.zeros([DIRECTIONS,2])
        c_train = obj['direction']
        rs_train = obj['rs']

        #true = mean + offset
        #offset = true - mean
        #pre = mean + offset_pre

        t1_train[c_train,:] = [np.cos(obj['theta1']),np.sin(obj['theta1'])]
        t2_train[c_train,:] = [np.cos(obj['theta2']),np.sin(obj['theta2'])]

        if c_train == 0:
            c_train_flip = 1
        else:
            c_train_flip = 0

        t1_train_flip = np.zeros([DIRECTIONS,2])
        t2_train_flip = np.zeros([DIRECTIONS,2])
        rs_train_flip = 1.0 - obj['rs']
        rs_train_flip = rs_train_flip - rs_mean[c_train_flip]

        t1_train_flip[c_train_flip,:] = [np.cos(obj['theta2']),np.sin(obj['theta2'])]
        t2_train_flip[c_train_flip,:] = [np.cos(obj['theta1']),np.sin(obj['theta1'])]

        obj['train'] = {
            'rs':rs_train,
            'theta1':t1_train,
            'theta2':t2_train,
            'confidence':np.array([1,0]) if c_train==0 else np.array([0,1])
        }
        obj['train_flip'] = {
            'rs':rs_train_flip,
            'theta1':t1_train_flip,
            'theta2':t2_train_flip,
            'confidence':np.array([1,0]) if c_train_flip==0 else np.array([0,1])
        }
    return all_objs

def prepare_input_and_output(train_inst):
    xmin = train_inst['xmin']
    ymin = train_inst['ymin']
    xmax = train_inst['xmax']
    ymax = train_inst['ymax']
    img = Image.open(train_inst['image'])
    img = img.convert('RGB')
    img_r = img.crop((xmin, ymin, xmax, ymax))
    img_r = img_r.resize((H, W))
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
    img_r = np.array(img_r).astype(np.float32)
    img_r = img_r - np.array([[[103.939, 116.779, 123.68]]])
    if flip > 0.5:
        return img_r, train_inst['train_flip']['rs'],train_inst['train_flip']['theta1'],train_inst['train_flip']['theta2'],train_inst['train_flip']['confidence']
    else:
        return img_r, train_inst['train']['rs'],train_inst['train']['theta1'],train_inst['train']['theta2'],train_inst['train']['confidence']


def data_gen(all_objs, batch_size):
    num_obj = len(all_objs)
    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj
    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        currt_inst = 0
        img_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
        rs_batch = np.zeros((r_bound - l_bound, DIRECTIONS))
        theta1_batch = np.zeros((r_bound - l_bound, DIRECTIONS,2))
        theta2_batch = np.zeros((r_bound - l_bound, DIRECTIONS,2))
        confidence_batch = np.zeros((r_bound - l_bound, DIRECTIONS))

        for key in keys[l_bound:r_bound]:
            image, rs, theta1, theta2, confidence = prepare_input_and_output(all_objs[key])
            img_batch[currt_inst, :] = image
            rs_batch[currt_inst, :] = rs
            theta1_batch[currt_inst, :] = theta1
            theta2_batch[currt_inst, :] = theta2
            confidence_batch[currt_inst, :] = confidence

            currt_inst += 1

        yield img_batch, [rs_batch, theta1_batch, theta2_batch,confidence_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj: r_bound = num_obj
























