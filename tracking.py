import cv2 as cv
import numpy as np
import random
from scipy.optimize import linear_sum_assignment



img_dir = 'videos'
box_dir = 'videos_2d_box'
#box_dir = 'videos_2d_refine'
para_dir = 'videos_result'
#para_dir = 'videos_result_refine'
output_dir = 'videos_3d_box'

dt = 1.
alpha = 0.85
beta = 0.05

def read_3d(filename):
    try:
        f = open(filename,'r')
    except:
        return None
    lines = f.readlines()
    f.close()
    paras = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) < 3:
            continue
        new_para = [float(line[0]),float(line[1]),float(line[2])]
        #if new_para[2] > 0.15:
        #    new_para[2] *= 0.8
        paras.append(new_para)

    return paras


def read_2d(filename):
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
        boxes.append([int(line[0]),int(line[1]),int(line[2]),int(line[3])])
    return boxes

def compute_mea(boxes):
    meas = []
    for i in range(boxes.shape[0]):
        meas.append([(boxes[i,0]+boxes[i,2])/2.,(boxes[i,1]+boxes[i,3])/2.])
        #meas.append([boxes[i,0],boxes[i,1]])
    return np.array(meas)


def alpha_beta_tracking(xk,vk,para,boxes,can_mea,mea_para,mea_boxes):
    _xk_1 = xk + dt * vk
    _vk_1 = vk
    zk_1, new_x, wl, z_para, new_para, z_boxes, new_boxes= gnnsf_opt(_xk_1,can_mea,mea_para,mea_boxes)
    xk_1 = np.delete(_xk_1,wl,axis=0)
    vk_1 = np.delete(_vk_1,wl,axis=0)
    para_1 = np.delete(para,wl,axis=0)
    para_1 = 0.8 * para_1 + 0.2 * z_para
    boxes_1 = z_boxes
    rk_1 = zk_1 - xk_1
    xk_1 = xk_1 + alpha * rk_1
    vk_1 = vk_1 + (beta/dt) * rk_1
    return xk_1,vk_1,new_x,wl, para_1, new_para, boxes_1, new_boxes


def gnnsf_opt(_xk_1,can_mea,mea_para,mea_boxes):
    n = _xk_1.shape[0]
    m = can_mea.shape[0]
    pos = _xk_1
    p = np.expand_dims(pos, axis=1)
    p = np.repeat(p, m, axis=1)
    mea = np.expand_dims(can_mea, axis=1)
    mea = np.repeat(mea, n, axis=1)
    mea = mea.transpose((1, 0, 2))
    s = p - mea
    s = np.square(s[:, :, 0]) + np.square(s[:, :, 1])

    z = np.zeros_like(pos)
    z_para = np.zeros([n, 3])
    z_boxes = np.zeros([n, 4])

    if n > m:
        inf_vec = np.ones([n,1]) * 999999.0
        for _ in range(n-m):
            s = np.concatenate((s,inf_vec),axis=1)
        row_ind, col_ind = linear_sum_assignment(s)
        wl = []
        for ri in row_ind:
            ci = col_ind[ri]
            if ci < m:
                z[ri,:] = can_mea[ci, :]
                z_para[ri,:] = mea_para[ci, :]
                z_boxes[ri,:] = mea_boxes[ci,:]
            else:
                wl.append(ri)
        z = np.delete(z, wl, axis=0)
        z_para = np.delete(z_para, wl, axis=0)
        z_boxes = np.delete(z_boxes, wl, axis=0)
        new_x = None
        new_para = None
        new_boxes = None
        return z, new_x, wl, z_para, new_para, z_boxes, new_boxes

    elif n < m:
        inf_vec = np.ones([1,m]) * 9999999.0
        for _ in range(m-n):
            s = np.concatenate((s,inf_vec),axis=0)
        row_ind, col_ind = linear_sum_assignment(s)
        new_l = []
        for ri in row_ind:
            ci = col_ind[ri]
            if ri < n:
                z[ri, :] = can_mea[ci, :]
                z_para[ri, :] = mea_para[ci, :]
                z_boxes[ri, :] = mea_boxes[ci, :]
            else:
                new_l.append(ci)
        new_x = can_mea[new_l]
        new_para = mea_para[new_l]
        new_boxes = mea_boxes[new_l]
        wl = []
        return z, new_x, wl, z_para, new_para, z_boxes, new_boxes
    else:
        row_ind, col_ind = linear_sum_assignment(s)
        for ri in row_ind:
            ci = col_ind[ri]
            z[ri, :] = can_mea[ci, :]
            z_para[ri, :] = mea_para[ci, :]
            z_boxes[ri, :] = mea_boxes[ci, :]
        wl = []
        new_x = None
        new_para = None
        new_boxes = None
        return z, new_x, wl, z_para, new_para, z_boxes, new_boxes




def gnnsf(_xk_1,can_mea,mea_para,mea_boxes):
    n = _xk_1.shape[0]
    m = can_mea.shape[0]
    pos = _xk_1
    p = np.expand_dims(pos, axis=1)
    p = np.repeat(p, m, axis=1)
    mea = np.expand_dims(can_mea, axis=1)
    mea = np.repeat(mea, n, axis=1)
    mea = mea.transpose((1, 0, 2))
    s = p - mea
    s = np.square(s[:, :, 0]) + np.square(s[:, :, 1])
    mark = np.ones(m)
    mmap = np.ones(n) * -1
    z = np.zeros_like(pos)
    z_para = np.zeros([n,3])
    z_boxes = np.zeros([n,4])
    for i in range(n):
        min_dis = 999999
        min_arg = -1
        for j in range(m):
            if mark[j] == 0:
                continue
            if s[i, j] < min_dis:
                min_dis = s[i, j]
                min_arg = j
        if min_arg != -1:
            mark[min_arg] = 0
            mmap[i] = min_arg
            z[i, 0] = can_mea[min_arg, 0]
            z[i, 1] = can_mea[min_arg, 1]
            z_para[i,:] = mea_para[min_arg,:]
            z_boxes[i,:] = mea_boxes[min_arg,:]
    wl = np.where(mmap == -1)
    if n < m:
        new_x = can_mea[np.where(mark == 1)]
        new_para = mea_para[np.where(mark == 1)]
        new_boxes = mea_boxes[np.where(mark == 1)]
    elif n > m:
        new_x = None
        new_para = None
        new_boxes = None
        z = np.delete(z, wl, axis=0)
        z_para = np.delete(z_para,wl,axis=0)
        z_boxes = np.delete(z_boxes,wl,axis=0)
    else:
        new_x = None
        new_para = None
        new_boxes = None
    return z, new_x, wl, z_para, new_para, z_boxes, new_boxes




def tracking():
    for img_index in range(420):
        print(img_index)
        img_file = img_dir + '/' + '%d.png'%img_index
        box_file = box_dir + '/' + '%d.txt'%img_index
        para_file = para_dir + '/' + '%d.txt'%img_index
        out_file = output_dir + '/' + '%d.png'%img_index

        img = cv.imread(img_file)
        mea_boxes = read_2d(box_file)
        mea_paras = read_3d(para_file)

        mea_boxes = np.array(mea_boxes)
        mea_paras = np.array(mea_paras)

        if img_index == 0:
            xk = compute_mea(mea_boxes)
            vk = np.zeros_like(xk)
            box_paras = mea_paras
            boxes = mea_boxes
            colors = []
            for _ in range(xk.shape[0]):
                colors.append([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
            colors = np.array(colors)
            continue

        mea_pos = compute_mea(mea_boxes)

        xk_1, vk_1, new_x, wl, box_paras_1, new_para, boxes_1, new_boxes = alpha_beta_tracking(xk, vk, box_paras,boxes,mea_pos,mea_paras,mea_boxes)
        colors = np.delete(colors,wl,axis=0)
        if new_x is not None:
            xk_1 = np.concatenate((xk_1, new_x), axis=0)
            box_paras_1 = np.concatenate((box_paras_1,new_para),axis=0)
            boxes_1 = np.concatenate((boxes_1,new_boxes),axis=0)
            new_v = np.zeros_like(new_x)
            vk_1 = np.concatenate((vk_1, new_v), axis=0)
            new_color = []
            for _ in range(new_x.shape[0]):
                new_color.append([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
            new_color = np.array(new_color)
            colors = np.concatenate((colors,new_color),axis=0)

        xk = xk_1
        vk = vk_1
        box_paras = box_paras_1
        boxes = boxes_1
        visualization(img,boxes,box_paras,colors)
        cv.imwrite(out_file,img)

def visualization(img,boxes,paras,colors):
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

    def draw_3d_box(img, bvs, stx, sty,col):
        #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        vs = []
        for var in bvs:
            vs.append((int(stx + var[0]), int(sty + var[1])))
        #for i in range(8):
        #    cv.putText(img,'%d'%i,vs[i] ,cv.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

        color = (int(col[0]),int(col[1]),int(col[2]))

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


    ###############################################

    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3]
        rs, tan_theta1, tan_theta2 = paras[i,0], paras[i,1], paras[i,2]
        vs = compute_box(rs,tan_theta1,tan_theta2,xmax-xmin,ymax-ymin)
        draw_3d_box(img,vs,xmin,ymin,colors[i,:])
    ###############################################




def video_play():
    while True:
        for img_index in range(420):
            img_file = output_dir + '/' + '%d.png' % img_index
            img = cv.imread(img_file)
            cv.imshow('video',img)
            cv.waitKey(100)






if __name__ == '__main__':





    tracking()
    video_play()


















