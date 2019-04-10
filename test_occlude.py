import numpy as np
import glob
import random
import cv2 as cv
import matplotlib.pyplot as plt

VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']


def read_results(filename,cali_file):
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    data = []
    catagory = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0 or line[0] not in VEHICLES:
            continue
        catagory.append(line[0])
        n_data = line[4:15]
        new_data = [float(var) for var in n_data]
        data.append(new_data)
    f1 = open(cali_file,'r')
    lines = f1.readlines()
    f1.close()
    p_line = lines[2]
    row_data = p_line.strip().split(' ')[1:]
    cali_mat = np.array([float(var) for var in row_data]).reshape([3,4])
    return data, cali_mat,catagory



def compute_box(data,cali_mat,kind):
    xmin, ymin, xmax, ymax = data[0], data[1], data[2], data[3]
    box_2d = [(int(xmin),int(ymin)),(int(xmax),int(ymax))]
    dz, dy, dx = data[4], data[5], data[6]
    px, py, pz = data[7], data[8], data[9]
    theta = data[10]
    #convert to [0,2pi)
    if theta < 0:
        theta = theta + np.pi * 2.0
    theta = (theta+np.pi) % (np.pi * 2.0)
    coor_rect = []
    if kind in ['Car', 'Truck', 'Van', 'Tram']:
        coor_rect.append([dx/2.0,0,dz/2.0])
        coor_rect.append([dx/2.0,0,-1*dz/2.0])
        coor_rect.append([-1*dx/2.0,0,-1*dz/2.0])
        coor_rect.append([-1*dx/2.0,0,dz/2.0])
        coor_rect.append([dx/2.0,-1*dy,dz/2.0])
        coor_rect.append([dx/2.0,-1*dy,-1*dz/2.0])
        coor_rect.append([-1*dx/2.0,-1*dy,-1*dz/2.0])
        coor_rect.append([-1*dx/2.0,-1*dy,dz/2.0])
    else:
        coor_rect.append([dx / 2.0, 0, dy / 2.0])
        coor_rect.append([dx / 2.0, 0, -1 * dy / 2.0])
        coor_rect.append([-1 * dx / 2.0, 0, -1 * dy / 2.0])
        coor_rect.append([-1 * dx / 2.0, 0, dy / 2.0])
        coor_rect.append([dx / 2.0, -1*dz, dy / 2.0])
        coor_rect.append([dx / 2.0, -1*dz, -1 * dy / 2.0])
        coor_rect.append([-1 * dx / 2.0, -1*dz, -1 * dy / 2.0])
        coor_rect.append([-1 * dx / 2.0, -1*dz, dy / 2.0])

    coor_rect = np.array(coor_rect)
    rot_mat = np.array([[np.cos(-1*theta),0,-1*np.sin(-1*theta)],[0.0,1.0,0.0],[np.sin(-1*theta),0,np.cos(-1*theta)]])
    r0 = np.array([px,py,pz])
    coor_cam = np.matmul(rot_mat,coor_rect.transpose()).transpose() + r0
    ones = np.ones([8,1],np.float)
    coor_cam = np.concatenate((coor_cam,ones),axis=1)
    coor_img = np.matmul(cali_mat,coor_cam.transpose()).transpose()
    coor_img = coor_img / coor_img[:,2].reshape([-1,1])
    coor_ret = coor_img[:,:2]
    return coor_ret, box_2d, pz


def compute_2d_box(coor,img_w,img_h):
    xs = coor[:,0]
    ys = coor[:,1]
    xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    xmin = max(0,xmin)
    ymin = max(0,ymin)
    xmax = min(xmax,img_w-1)
    ymax = min(ymax,img_h-1)
    return xmin, ymin, xmax, ymax


def compute_IOU(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1+width1, x2+width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx-startx)

    endy = max(y1+height1, y2+height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    return ratio


def compute_oou(boxes,zs,img_w,img_h):
    box_num = np.shape(boxes)[0]
    img_plain = np.zeros((img_h,img_w))
    oous = []
    for i in range(box_num):
        img_plain[boxes[i,1]:boxes[i,3],boxes[i,0]:boxes[i,2]] = 1.
        for j in range(box_num):
            if j != i and zs[j] < zs[i]:
                img_plain[boxes[j,1]:boxes[j,3],boxes[j,0]:boxes[j,2]] = 0.
        oous.append(1-np.sum(img_plain)/((boxes[i,3]-boxes[i,1])*(boxes[i,2]-boxes[i,0])))
        img_plain[:,:] = 0.
    return oous


def draw_2d_box(img,xmin,ymin,xmax,ymax):
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color,2)



def op_image(result_file,cali_file,img_file,out_path,img_out_path):
    box_datas, cali_mat,catagory = read_results(result_file,cali_file)
    img = cv.imread(img_file)
    img_w, img_h = img.shape[1], img.shape[0]
    boxes_2d = []
    zs = []
    for i in range(len(box_datas)):
        coor_img,_, pz = compute_box(box_datas[i],cali_mat,catagory[i])
        zs.append(pz)
        xmin, ymin, xmax, ymax = compute_2d_box(coor_img,img_w,img_h)
        boxes_2d.append([xmin, ymin, xmax, ymax])
        #draw_2d_box(img,xmin, ymin, xmax, ymax)
    boxes_2d = np.array(boxes_2d).astype(np.int)
    oous = compute_oou(boxes_2d,zs,img_w,img_h)
    #cv.imwrite(img_out_path,img)
    f = open(out_path,'w')
    for i in range(boxes_2d.shape[0]):
        f.write('%s %d %d %d %d %f %f %f %f %f %f\n'%(catagory[i],boxes_2d[i,0],boxes_2d[i,1],boxes_2d[i,2],boxes_2d[i,3],oous[i],box_datas[i][4],box_datas[i][5],box_datas[i][6],box_datas[i][10],box_datas[i][7]/box_datas[i][9]))
    f.close()








def test_all(label_dir,cali_dir,image_dir,output_dir,img_out_dir):
    labels = glob.glob(label_dir+'/*')
    for label in labels:
        label_name = label.split('/')[-1].split('.')[0]
        print(label_name)
        label_file = label
        cali_file = cali_dir + '/' + label_name + '.txt'
        out_file = output_dir + '/' + label_name + '.txt'
        img_file = image_dir + '/' + label_name + '.png'
        img_out_file = img_out_dir + '/' + label_name + '.png'
        op_image(label_file,cali_file,img_file,out_file,img_out_file)






if __name__ == '__main__':
    image_dir = '/Users/yangchenhongyi/Documents/3d_box/3D-Deepbox-master/deep_box_using/test_image1'
    label_dir = '/Users/yangchenhongyi/Documents/3d_box/3D-Deepbox-master/deep_box_using/test_label1'
    cali_dir = '/Users/yangchenhongyi/Documents/3d_box/data_object_calib/training/calib'
    out_dir = '/Users/yangchenhongyi/Documents/3d_box/3D-Deepbox-master/deep_box_using/occlude_label'
    img_out_dir = 'test_2dimg'
    test_all(label_dir,cali_dir,image_dir,out_dir,img_out_dir)











































