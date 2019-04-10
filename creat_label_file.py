import numpy as np
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
        if len(line) == 0 or line[0] not in VEHICLES or int(line[2]) == 2:
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
    coor_ret = np.array(amend_box2(coor_ret,box_2d))
    return coor_ret, box_2d

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
    delta_1 = ymax - coor_img[pos_dict1[key][0], 1]
    coor_img[pos_dict1[key][0], 1] += delta_1
    coor_img[pos_dict1[key][1], 1] += delta_1
    coor_img[pos_dict1[key][2], 1] += delta_1

    delta_2 = xmin - coor_img[argsort_x[0], 0]
    coor_img[argsort_x[0], 0] += delta_2
    coor_img[pos_dict1[key][0], 0] += delta_2
    # coor_img[pos_dict1[key][3], 0] += delta_2

    coor_img[argsort_x[-1], 0] = xmax
    coor_img[pos_dict1[key][3], 0] = coor_img[pos_dict1[key][2], 0] + coor_img[pos_dict1[key][1], 0] - coor_img[
        pos_dict1[key][0], 0]
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

def compute_para(coor,box_2d):
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    rs = (coor[7,0]-xmin) / (xmax-xmin)
    if rs == 0:
        theta1 = 0
    else:
        theta1 = np.arctan((ymax-coor[4,1])/(rs*(xmax-xmin)))
    if rs == 1.0:
        theta2 = 0
    else:
        theta2 = np.arctan((ymax-coor[6,1])/((1-rs)*(xmax-xmin)))
    if coor[7,0] > coor[5,0]:
        kind = 1
    else:
        kind = 0
    return rs, theta1, theta2, kind

def op_file(in_file,cali_file,out_file):
    data,cali_mat,catagories = read_results(in_file,cali_file)
    f = open(out_file,'w')
    for i in range(len(data)):
        coor_img, box_2d = compute_box(data[i], cali_mat, catagories[i])
        rs, theta1, theta2, kind = compute_para(coor_img,box_2d)
        f.write('%s %d %d %d %d %f %f %f %d\n'%(catagories[i],box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1],rs,theta1,theta2,kind))
    f.close()

def op_all(label_dir,cali_dir,output_dir):
    file_list = glob.glob(label_dir+'/*')
    for label_file in file_list:
        name = label_file.split('/')[-1]
        cali_file = cali_dir + '/' + name
        out_file = output_dir + '/' + name
        op_file(label_file,cali_file,out_file)

if __name__ == '__main__':
    label_dir = '/Users/yangchenhongyi/Documents/3d_box/training/label_2'
    cali_dir = '/Users/yangchenhongyi/Documents/3d_box/data_object_calib/training/calib'
    output_dir = '/Users/yangchenhongyi/Documents/3d_box/new_label'
    op_all(label_dir,cali_dir,output_dir)



















