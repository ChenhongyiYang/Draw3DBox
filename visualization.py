import numpy as np
import random
import cv2 as cv
import glob


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
    #coor_ret = np.array(amend_box2(coor_ret,box_2d))
    return coor_ret, box_2d,theta



def amend_box2(coor,box_2d):
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    coor_img = coor.copy()
    pos_dict = {3:[3,2,0,7],0:[0,1,3,4],1:[1,0,2,5],2:[2,3,1,6]}
    pos_dict1 = {3:[3,2,0,5],0:[0,1,3,6],1:[1,0,2,7],2:[2,3,1,4]}
    bottom_coor = coor_img[:4,:]
    y = bottom_coor[:,1]
    argsort_y = np.argsort(y)
    x = bottom_coor[:,0]
    argsort_x = np.argsort(x)
    argsort_x =argsort_x.tolist()
    if argsort_x.index(argsort_y[2]) != 0 and argsort_x.index(argsort_y[2]) != 3:
        key = argsort_y[2]
    else:
        key = argsort_y[3]
    vs = np.zeros_like(coor_img)

    '''
    ###################

    delta_1 = ymax - coor_img[pos_dict[key][0], 1]
    coor_img[pos_dict[key][0], 1] += delta_1
    coor_img[pos_dict[key][1], 1] += delta_1
    coor_img[pos_dict[key][2], 1] += delta_1
    delta_2 = xmin - coor_img[argsort_x[0], 0]
    coor_img[argsort_x[0], 0] += delta_2
    coor_img[pos_dict[key][0], 0] += delta_2
    coor_img[pos_dict[key][3], 0] += delta_2

    coor_img[argsort_x[-1], 0] = xmax
    ############
    vs[7, :] = coor_img[pos_dict[key][0], :]
    vs[6, :] = coor_img[pos_dict[key][1], :]
    vs[4, :] = coor_img[pos_dict[key][2], :]
    vs[3, :] = coor_img[pos_dict[key][3], :]
    vs[0, :] = vs[4, :] + vs[3, :] - vs[7, :]
    vs[2, :] = vs[3, :] + vs[6, :] - vs[7, :]
    vs[1, :] = vs[0, :] + vs[2, :] - vs[3, :]
    vs[5, :] = vs[4, :] + vs[1, :] - vs[0, :]

    ##################
    '''

    delta_1 = ymax - coor_img[pos_dict1[key][0], 1]
    coor_img[pos_dict1[key][0], 1] += delta_1
    coor_img[pos_dict1[key][1], 1] += delta_1
    coor_img[pos_dict1[key][2], 1] += delta_1

    delta_2 = xmin - coor_img[argsort_x[0], 0]
    coor_img[argsort_x[0], 0] += delta_2
    coor_img[pos_dict1[key][0], 0] += delta_2
    #coor_img[pos_dict1[key][3], 0] += delta_2

    coor_img[argsort_x[-1], 0] = xmax
    coor_img[pos_dict1[key][3], 0] = coor_img[pos_dict1[key][2],0] + coor_img[pos_dict1[key][1],0] - coor_img[pos_dict1[key][0],0]
    coor_img[pos_dict1[key][3], 1] = ymin
    ##################

    vs[7, :] = coor_img[pos_dict1[key][0], :]
    vs[6, :] = coor_img[pos_dict1[key][1], :]
    vs[4, :] = coor_img[pos_dict1[key][2], :]
    vs[1, :] = coor_img[pos_dict1[key][3], :]
    vs[0, :] = vs[1, :] + vs[7, :] - vs[6, :]
    vs[3, :] = vs[7, :] + vs[0, :] - vs[4, :]
    vs[2, :] = vs[6, :] + vs[0, :] - vs[4, :]
    vs[5, :] = vs[1, :] + vs[6, :] - vs[2, :]
    return vs

def amend_box1(theta,coor_img,box_2d):
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    w, h = xmax - xmin, ymax - ymin
    coor = np.zeros_like(coor_img)
    if theta < np.pi / 2.0 or (theta > np.pi and theta < np.pi * 1.5):

        if theta < np.pi / 2.0:
            coor[1, :] = coor_img[5, :]
            coor[5, :] = coor_img[1, :]
            coor[2, :] = coor_img[4, :]
            coor[6, :] = coor_img[0, :]
            coor[0, :] = coor_img[6, :]
            coor[4, :] = coor_img[2, :]
            coor[3, :] = coor_img[7, :]
            coor[7, :] = coor_img[3, :]

        else:
            coor[1, :] = coor_img[7, :]
            coor[5, :] = coor_img[3, :]
            coor[2, :] = coor_img[6, :]
            coor[6, :] = coor_img[2, :]
            coor[0, :] = coor_img[4, :]
            coor[4, :] = coor_img[0, :]
            coor[3, :] = coor_img[5, :]
            coor[7, :] = coor_img[1, :]
    else:

        if theta > np.pi/2.0:
            coor[1, :] = coor_img[4, :]
            coor[5, :] = coor_img[0, :]
            coor[2, :] = coor_img[7, :]
            coor[6, :] = coor_img[3, :]
            coor[0, :] = coor_img[5, :]
            coor[4, :] = coor_img[1, :]
            coor[3, :] = coor_img[6, :]
            coor[7, :] = coor_img[2, :]
        else:
            coor[1, :] = coor_img[6, :]
            coor[5, :] = coor_img[2, :]
            coor[2, :] = coor_img[7, :]
            coor[6, :] = coor_img[3, :]
            coor[0, :] = coor_img[5, :]
            coor[4, :] = coor_img[1, :]
            coor[3, :] = coor_img[4, :]
            coor[7, :] = coor_img[0, :]
    vs = np.zeros_like(coor)
    vs[3,:] = coor[3,:]
    vs[4,:] = coor[4,:]
    vs[6,:] = coor[6,:]
    vs[7,:] = coor[7,:]
    vs[0,:] = vs[4,:] + vs[3,:] - vs[7,:]
    vs[2,:] = vs[3,:] + vs[6,:] - vs[7,:]
    vs[1,:] = vs[0,:] + vs[2,:] - vs[3,:]
    vs[5,:] = vs[4,:] + vs[1,:] - vs[0,:]
    return vs


def amend_box(theta,coor_img,box_2d):
    vs = []
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    w, h = xmax - xmin, ymax - ymin
    coor = np.zeros_like(coor_img)
    if theta < np.pi/2.0 or (theta > np.pi and theta < np.pi * 1.5):

        if theta < np.pi/2.0:
            coor[1,:] = coor_img[1,:]
            coor[5,:] = coor_img[5,:]
            coor[2,:] = coor_img[0,:]
            coor[6,:] = coor_img[4,:]
            coor[0,:] = coor_img[2,:]
            coor[4,:] = coor_img[6,:]
            coor[3,:] = coor_img[1,:]
            coor[7,:] = coor_img[5,:]

        else:
            coor[1, :] = coor_img[3, :]
            coor[5, :] = coor_img[7, :]
            coor[2, :] = coor_img[2, :]
            coor[6, :] = coor_img[6, :]
            coor[0, :] = coor_img[0, :]
            coor[4, :] = coor_img[4, :]
            coor[3, :] = coor_img[1, :]
            coor[7, :] = coor_img[5, :]

        v7x = coor[7,0]
        s = v7x - xmin
        v5y = coor[5, 1]
        ds1 = ymax - v5y
        v6y = coor[4,1]
        ds2 = ds1 + v6y -ymax


        vs.append([0, max(ds1 - ds2, 0)])
        vs.append([w - s, 0])
        vs.append([w, ds2])
        vs.append([s, ds1])
        vs.append([0, h - ds2])
        vs.append([w - s, h - ds1])
        vs.append([w, min(h - ds1 + ds2, h)])
        vs.append([s, h])

    else:

        if theta > np.pi/2.0:
            coor[1,:] = coor_img[0,:]
            coor[5,:] = coor_img[4,:]
            coor[2,:] = coor_img[3,:]
            coor[6,:] = coor_img[7,:]
            coor[0,:] = coor_img[1,:]
            coor[4,:] = coor_img[5,:]
            coor[3,:] = coor_img[2,:]
            coor[7,:] = coor_img[6,:]
        else:
            coor[1, :] = coor_img[2, :]
            coor[5, :] = coor_img[6, :]
            coor[2, :] = coor_img[1, :]
            coor[6, :] = coor_img[5, :]
            coor[0, :] = coor_img[3, :]
            coor[4, :] = coor_img[7, :]
            coor[3, :] = coor_img[0, :]
            coor[7, :] = coor_img[4, :]

        v7x = coor[7,0]
        s = xmax - v7x
        v5y = coor[5,1]
        ds1 = ymax - v5y
        v4y = coor[4, 1]
        ds2 = ds1 + v4y -ymax

        vs.append([0, ds2])
        vs.append([s, 0])
        vs.append([w, max(ds1 - ds2, 0)])
        vs.append([w - s, ds1])
        vs.append([0, min(h - ds1 + ds2, h)])
        vs.append([s, h - ds1])
        vs.append([w, h - ds2])
        vs.append([w - s, h])
    rs, rs1, rs2 = s/w, ds1/h, ds2/h
    print(rs,rs1,rs2)

    for var in vs:
        var[0] += xmin
        var[1] += ymin
    return vs






def draw_box(img,img_coor,theta):
    #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
    if theta < np.pi/2.0 or (theta > np.pi and theta < np.pi * 1.5):

        if theta < np.pi/2.0:
            color = colors[0]
        else:
            color = colors[1]
    else:
        if theta < np.pi:
            color = colors[2]
        else:
            color = colors[3]
    #print(img_coor)
    vs = []

    for i in range(8):
        vs.append((int(img_coor[i,0]),int(img_coor[i,1])))

    for i in range(8):
        cv.putText(img,'%d'%i,vs[i] ,cv.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

    cv.line(img,vs[0],vs[1],color,2,8,0)
    cv.line(img,vs[1],vs[2],color,2,8,0)
    cv.line(img,vs[2],vs[3],color,2,8,0)
    cv.line(img,vs[3],vs[0],color,2,8,0)

    cv.line(img,vs[4],vs[5],color,2,8,0)
    cv.line(img,vs[5],vs[6],color,2,8,0)
    cv.line(img,vs[6],vs[7],color,2,8,0)
    cv.line(img,vs[7],vs[4],color,2,8,0)

    cv.line(img,vs[0],vs[4],color,2,8,0)
    cv.line(img,vs[3],vs[7],color,2,8,0)
    cv.line(img,vs[2],vs[6],color,2,8,0)
    cv.line(img,vs[1],vs[5],color,2,8,0)

def draw_box_2d(img,box_2d):
    color = (0,0,0)
    cv.rectangle(img,box_2d[0], box_2d[1],color,2)


def op_image(img_path,result_file,cali_file,out_path=None):
    img = cv.imread(img_path)
    print(img.shape)
    box_datas, cali_mat,catagory = read_results(result_file,cali_file)
    for i in range(len(box_datas)):
        print(catagory[i])
        coor_img,box_2d,theta = compute_box(box_datas[i],cali_mat,catagory[i])
        draw_box(img,coor_img,theta)
        draw_box_2d(img,box_2d)
    #img = cv.flip(img,1)
    cv.imwrite(out_path,img)


def draw_all(img_dir,label_dir,cali_dir,output_dir):
    images = glob.glob(img_dir+'/*')
    for image in images:
        image_name = image.split('/')[-1].split('.')[0]
        print(image_name)
        label_file = label_dir + '/' + image_name + '.txt'
        cali_file = cali_dir + '/' + image_name + '.txt'
        out_file = output_dir + '/' + image_name + '.png'
        op_image(image,label_file,cali_file,out_file)


if __name__ == '__main__':
    image_dir = 'test_image'
    label_dir = 'test_label'
    draw_dir = 'bounding_box_img'
    cali_dir = 'test_cali'
    result_dir = 'test_result'
    draw_all(image_dir, label_dir, cali_dir, draw_dir)





