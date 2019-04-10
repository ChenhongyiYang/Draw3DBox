import tensorflow as tf
import cv2 as cv
import numpy as np
import build_model
import os
import glob
import random


inputs = build_model.inputs

VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']

def read_2d_box(filename):
    try:
        f = open(filename, 'r')
    except:
        return None
    lines = f.readlines()
    f.close()
    boxes = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) < 4:
            continue
        boxes.append([(int(line[0]),int(line[1])),(int(line[2]),int(line[3]))])
    return boxes

def read_2d_box_2(filename):
    try:
        f = open(filename, 'r')
    except:
        return None
    lines = f.readlines()
    f.close()
    boxes = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0 or line[0] not in VEHICLES:
            continue
        boxes.append([(int(float(line[4])),int(float(line[5]))),(int(float(line[6])),int(float(line[7])))])
    return boxes



def test(box_dir,img_dir,out_dir,model):
    rs, theta1, theta2, probability, _, _, _, _, _, _ = build_model.build_model()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model)


    image_list = glob.glob(img_dir+'/*')
    for image in image_list:
        image_name = image.split('/')[-1].split('.')[0]
        print(image_name)
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        boxes_2d = read_2d_box(box_dir+'/'+image_name+'.txt')
        if boxes_2d is None:
            continue

        box_num = len(boxes_2d)
        print(boxes_2d)
        img_tensor = np.zeros([box_num,224,224,3])
        for i in range(box_num):
            xmin, ymin, xmax, ymax = boxes_2d[i][0][0], boxes_2d[i][0][1], boxes_2d[i][1][0], boxes_2d[i][1][1]
            img_croped = img[ymin:ymax,xmin:xmax,:]
            img_input = img_croped.copy()
            img_input = cv.resize(img_input,(224,224))
            img_input = img_input
            img_tensor[i,:] = img_input

        _rs, _theta1,_theta2,_prob = sess.run([rs, theta1, theta2, probability],feed_dict={inputs:img_tensor})

        fo = open(out_dir+'/'+image_name+'.txt','w')
        for i in range(box_num):
            d = np.argmax(_prob[i,:])
            rs_o = _rs[i,d]
            tan_theta1 = _theta1[i,d,1] / _theta1[i,d,0]
            tan_theta2 = _theta2[i,d,1] / _theta2[i,d,0]
            fo.write('%f %f %f\n'%(rs_o,tan_theta1,tan_theta2))
        fo.close()


def visualization(box_dir,result_dir,image_dir,output_dir):
    def compute_box(rs, tan_theta1, tan_theta2, w, h):
        vs = np.zeros([8, 2])
        vs[7, 0] = rs * w
        vs[7, 1] = h
        vs[4, 0] = 0
        vs[4, 1] = h - tan_theta1 * (w * rs)
        vs[6, 0] = w
        vs[6, 1] = h - (w - w * rs) * tan_theta2
        vs[1, 0] = vs[4, 0] + vs[6, 0] - vs[7, 0]
        vs[1, 1] = 0
        vs[0, :] = vs[1, :] + vs[7, :] - vs[6, :]
        vs[3, :] = vs[7, :] + vs[0, :] - vs[4, :]
        vs[2, :] = vs[6, :] + vs[0, :] - vs[4, :]
        vs[5, :] = vs[1, :] + vs[6, :] - vs[2, :]
        return vs

    def draw_3d_box(img, bvs, stx, sty):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        vs = []
        for var in bvs:
            vs.append((int(stx + var[0]), int(sty + var[1])))
        #for i in range(8):
        #    cv.putText(img,'%d'%i,vs[i] ,cv.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

        cv.line(img, vs[0], vs[1], color, 2, 8, 0)
        cv.line(img, vs[1], vs[2], color, 2, 8, 0)
        cv.line(img, vs[2], vs[3], color, 2, 8, 0)
        cv.line(img, vs[3], vs[0], color, 2, 8, 0)

        cv.line(img, vs[4], vs[5], color, 2, 8, 0)
        cv.line(img, vs[5], vs[6], color, 2, 8, 0)
        cv.line(img, vs[6], vs[7], color, 2, 8, 0)
        cv.line(img, vs[7], vs[4], color, 2, 8, 0)

        cv.line(img, vs[0], vs[4], color, 2, 8, 0)
        cv.line(img, vs[3], vs[7], color, 2, 8, 0)
        cv.line(img, vs[2], vs[6], color, 2, 8, 0)
        cv.line(img, vs[1], vs[5], color, 2, 8, 0)

    image_list = glob.glob(image_dir+'/*')
    for image in image_list:
        image_name = image.split('/')[-1].split('.')[0]
        print(image_name)
        img = cv.imread(image)
        boxes_2d = read_2d_box(box_dir + '/' + image_name + '.txt')
        if boxes_2d is None:
            continue
        box_num = len(boxes_2d)
        fr = open(result_dir + '/' + image_name + '.txt')
        result_lines = fr.readlines()
        fr.close()
        for i in range(box_num):
            xmin, ymin, xmax, ymax = boxes_2d[i][0][0], boxes_2d[i][0][1], boxes_2d[i][1][0], boxes_2d[i][1][1]
            line = result_lines[i].split(' ')
            rs ,tan_theta1, tan_theta2 = float(line[0]), float(line[1]), float(line[2])
            vs = compute_box(rs,tan_theta1,tan_theta2,xmax-xmin,ymax-ymin)
            draw_3d_box(img,vs,xmin,ymin)
        cv.imwrite(output_dir + '/' + image_name + '.png',img)





if __name__ == '__main__':
    model = 'model/model-5'
    image_dir = 'videos'
    result_dir = 'videos_result'
    box_2d_dir = 'videos_2d_box'
    box_3d_dir = 'videos_3d_box'

    if os.path.isdir(result_dir) == False:
        os.mkdir(result_dir)
    if os.path.isdir(box_3d_dir) == False:
        os.mkdir(box_3d_dir)





    test(box_2d_dir,image_dir,result_dir,model)
    #visualization(box_2d_dir,result_dir,image_dir,box_3d_dir)













































