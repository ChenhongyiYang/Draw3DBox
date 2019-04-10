import cv2 as cv
import numpy as np

box_dir = 'videos_2d_box'
refine_dir = 'videos_2d_refine'
img_dir = 'videos'

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

def compute_center(box):
    cen_x = (box[0] + box[2]) / 2.
    cen_y = (box[1] + box[3]) / 2.
    return cen_x, cen_y

def compute_area(box):
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    return area





def refine():
    for k in range(0,420):
        now_file = box_dir + '/' + '%d.txt' % k
        last_file = box_dir + '/' + '%d.txt' % (k-1)
        next_file = box_dir + '/' + '%d.txt' % (k+1)

        refine_file = refine_dir + '/' + '%d.txt' % k

        now_boxes = read_2d(now_file)
        _now_boxes = now_boxes[:]

        if k == 0 or k == 419:
            fw = open(refine_file, 'w')
            for box in _now_boxes:
                fw.write('%d %d %d %d\n' % (box[0], box[1], box[2], box[3]))
            fw.close()
            continue

        last_boxes = read_2d(last_file)
        next_boxes = read_2d(next_file)

        now_n = len(now_boxes)
        last_n = len(last_boxes)
        next_n = len(next_boxes)


        _new = False

        for i in range(last_n):
            last_area = compute_area(last_boxes[i])
            last_cent_x, last_cent_y = compute_center(last_boxes[i])
            find = False
            for j in range(now_n):
                now_area = compute_area(now_boxes[j])
                now_cent_x, now_cent_y = compute_center(now_boxes[j])
                d_cent = abs(now_cent_x - last_cent_x) + abs(now_cent_y - last_cent_y)
                d_area = abs(now_area - last_area) / last_area
                if d_cent < 15 or d_area < 0.1:
                    find = True
                    break

            if not find:
                for j in range(next_n):
                    next_area = compute_area(next_boxes[j])
                    next_cent_x, next_cent_y = compute_center(next_boxes[j])
                    d_cent = abs(next_cent_x - last_cent_x) + abs(next_cent_y - last_cent_y)
                    d_area = abs(last_area - next_area) / last_area

                    if d_cent < 20 and d_area < 0.10:
                        _now_boxes.append([(last_boxes[i][0]+next_boxes[j][0])//2, (last_boxes[i][1]+next_boxes[j][1])//2, (last_boxes[i][2]+next_boxes[j][2])//2, (last_boxes[i][3]+next_boxes[j][3])//2])
                        break

        fw = open(refine_file,'w')
        for box in _now_boxes:
            fw.write('%d %d %d %d\n'%(box[0],box[1],box[2],box[3]))
        fw.close()


def visualize():
    for i in range(420):
        img_file = img_dir + '/' + '%d.png' % i
        box_file = refine_dir + '/' + '%d.txt' % i

        img = cv.imread(img_file)
        boxes = read_2d(box_file)

        for box in boxes:
            cv.rectangle(img,(box[0],box[1]),(box[2],box[3]),color=(255,0,0),thickness=2)
        cv.imshow('video',img)
        cv.waitKey()


if __name__ == '__main__':
    #refine()
    visualize()















