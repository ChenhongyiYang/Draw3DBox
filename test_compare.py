import numpy as np
import random
import glob

VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']

def random_box(box_2d,w,h):
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    box_w, box_h = xmax-xmin, ymax-ymin
    xmin_range = [int(max(0,xmin-0.2*box_w)), int(min(w,xmin+0.2*box_w))]
    ymin_range = [int(max(0,ymin-0.2*box_h)), int(min(h,xmin+0.2*box_h))]

    _xmin = random.randint(xmin_range)
    _ymin = random.randint(ymin_range)

    xmax_range = [int(max(0,xmax-0.2*box_w)), int(min(w,xmax+0.2*box_w))]
    ymax_range = [int(max(0,ymax-0.2*box_h)), int(min(h,ymax+0.2*box_h))]

    _xmax = random.randint(xmax_range)
    _ymax = random.randint(ymax_range)

    return [(_xmin,_ymin),(_xmax,_ymax)]



def read_label(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    boxes = []
    centroids = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0 or line[0] not in VEHICLES:
            continue
        boxes.append([(int(float(line[4])),int(float(line[5]))),(int(float(line[6])),int(float(line[7])))])
        centroids.append([float(line[11]),float(line[12]),float(line[13])])
    return boxes, centroids



def read_cali(cali_file):
    f1 = open(cali_file, 'r')
    lines = f1.readlines()
    f1.close()
    p_line = lines[2]
    row_data = p_line.strip().split(' ')[1:]
    cali_mat = np.array([float(var) for var in row_data]).reshape([3, 4])
    return cali_mat


def read_result(result_file):
    f = open(result_file, 'r')
    lines = f.readlines()
    rs_list = []
    tan_t1_list = []
    tan_t2_list = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0:
            continue
        rs_list.append(float(line[0]))
        tan_t1_list.append(float(line[1]))
        tan_t2_list.append(float(line[2]))
    return rs_list, tan_t1_list, tan_t2_list




def compute_s(p_mat, centroid):
    cen_coor = np.array([[centroid[0],centroid[1],centroid[2],1]]).transpose()
    pt = np.matmul(p_mat,cen_coor)
    s = pt[2,0]
    return s

def compute_3d_box(rs,tan_t1,tan_t2,box_2d,s,P):
    xmin, ymin, xmax, ymax = box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]
    w, h = xmax-xmin, ymax-ymin

    v7 = [rs*w+xmin, ymax]
    v4 = [xmin, ymax - tan_t1 * (w * rs)]
    v6 = [xmax, ymax - (w - w * rs) * tan_t2]
    v1 = [v4[0] + v6[0] - v7[0], ymin]

    coor_img = np.array([v7,v4,v6,v1])
    coor_img = np.concatenate((coor_img,np.ones([4,1])),axis=1)
    coor_img = coor_img * s
    P1 = P[:,:3]
    P2 = P[:,3]
    coor_cam = np.linalg.inv(P1).dot((coor_img - P2).transpose())
    coor_cam = coor_cam.transpose()
    return coor_cam


def compute_dim(coor_cam):
    print(coor_cam)
    plain_pt = coor_cam[:3,:]
    h_pt = coor_cam[3,:]

    mA = plain_pt.copy()
    mA[:,0] = 1.
    A = np.linalg.det(mA)

    mB = plain_pt.copy()
    mB[:,1] = 1.
    B = np.linalg.det(mB)

    mC = plain_pt.copy()
    mC[:,2] = 1.
    C = np.linalg.det(mC)

    mD = plain_pt.copy()
    D = -1. * np.linalg.det(mD)

    height = abs(A*h_pt[0]+B*h_pt[1]+C*h_pt[2]+D) / np.sqrt(A**2+B**2+C**2)
    l1 = np.sqrt(np.sum(np.square(plain_pt[0,:] - plain_pt[1,:])))
    l2 = np.sqrt(np.sum(np.square(plain_pt[0,:] - plain_pt[2,:])))
    if l1 > l2:
        pl = plain_pt[1,:]
        pw = plain_pt[2,:]
        length = l1
        width = l2
    else:
        pl = plain_pt[2, :]
        pw = plain_pt[1, :]
        length = l2
        width = l1
    alpha = np.arctan(plain_pt[0,2]-pl[2]/plain_pt[0,0]-pl[0])
    return length, width, height, alpha



def compute_all(result_dir,cali_dir,label_dir,out_dir):
    result_list = glob.glob(result_dir + '/*')
    for result_file in result_list:
        result_name = result_file.split('/')[-1]
        print(result_name)
        rs_list, tan_t1_list, tan_t2_list = read_result(result_file)
        cali_mat = read_cali(cali_dir + '/' + result_name)
        boxes, centroids = read_label(label_dir+'/'+result_name)
        box_num = len(boxes)
        f = open(out_dir+'/'+result_name,'w')
        for i in range(box_num):
            s = compute_s(cali_mat,centroids[i])
            coor_cam = compute_3d_box(rs_list[i],tan_t1_list[i],tan_t2_list[i],boxes[i],s,cali_mat)
            length, width, height, alpha = compute_dim(coor_cam)
            f.write('%f %f %f %f\n'%(length, width, height, alpha))
        f.close()

if __name__ == '__main__':
    result_dir = 'test_result_2'
    cali_dir = 'test_cali'
    label_dir = 'test_label'
    out_dir = 'self_dim'
    compute_all(result_dir,cali_dir,label_dir,out_dir)












































