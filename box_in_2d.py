import numpy as np
import cv2 as cv





def compute_box(w,h,rs0,rs1,rs2):
    w = int(w)
    h = int(h)
    s = int(rs0*0.01 * w)
    ds1 = int(rs1*0.01 * h)
    ds2 = int(rs2*0.01 * h)

    vs1 = []
    vs1.append([0, max(ds1 - ds2,0)])
    vs1.append([w - s, 0])
    vs1.append([w, ds2])
    vs1.append([s, ds1])
    vs1.append([0, h - ds2])
    vs1.append([w - s, h - ds1])
    vs1.append([w, min(h - ds1 + ds2,h)])
    vs1.append([s, h])

    vs2 = []
    vs2.append([0, ds2])
    vs2.append([s, 0])
    vs2.append([w, max(ds1-ds2,0)])
    vs2.append([w-s, ds1])
    vs2.append([0, min(h - ds1 + ds2,h)])
    vs2.append([s, h - ds1])
    vs2.append([w, h - ds2])
    vs2.append([w-s, h])

    return vs1, vs2

def draw_arrow(img,bvs,stx,sty,w,h,method):
    vs = []
    for var in bvs:
        vs.append((int(stx + var[0]), int(sty + var[1])))

    if method == 0:
        st, end = 0,1
    elif method == 1:
        st,end = 2,1
    elif method == 2:
        st,end = 1,2
    else:
        st,end = 1,0

    ar_stx, ar_sty = stx+w//2, sty-h//4
    dx, dy = ar_stx-vs[st][0], ar_sty-vs[st][1]
    lx, ly = vs[end][0]-vs[st][0], vs[end][1]-vs[st][1]
    ori_l = np.sqrt(np.square(lx)+np.square(ly))
    max_l = np.sqrt(np.square(w)+np.square(h))
    lr = (h/4.0) /max_l
    #ar_endx = int((vs[end][0]+dx - ar_stx)*lr + ar_stx)
    #ar_endy = int((vs[end][1]+dy - ar_sty)*lr + ar_sty)
    ar_endx = int(lx*lr + ar_stx)
    ar_endy = int(ly*lr + ar_sty)
    cv.arrowedLine(img,(ar_stx,ar_sty),(ar_endx,ar_endy),(0,0,255),1)
    #print((ar_stx,ar_sty),(ar_endx,ar_endy))


def compute_box_2(rs,theta1,theta2,w,h):
    vs = np.zeros([8,2])
    vs[7,0] = rs * w
    vs[7,1] = h
    vs[4,0] = 0
    vs[4,1] = h - np.tan(theta1) * (w*rs)
    vs[6,0] = w
    vs[6,1] = h - (w-w*rs) * np.tan(theta2)
    vs[1,0] = vs[4,0] + vs[6,0] - vs[7,0]
    vs[1,1] = 0
    vs[0, :] = vs[1, :] + vs[7, :] - vs[6, :]
    vs[3, :] = vs[7, :] + vs[0, :] - vs[4, :]
    vs[2, :] = vs[6, :] + vs[0, :] - vs[4, :]
    vs[5, :] = vs[1, :] + vs[6, :] - vs[2, :]
    return vs


def draw_3d_box(img,bvs,stx,sty,method):
    col1 = (0,0,255)
    col2 = (255,0,0)

    if method == 0:
        cls = [col1,col1,col1,col1,col2,col2,col1,col1,col1,col1,col1,col2]
    else:
        cls = [col1, col1, col2, col2, col1, col1, col1, col1, col1, col2, col1, col1]
    vs = []
    for var in bvs:
        vs.append((int(stx+var[0]),int(sty+var[1])))

    for i in range(8):
        cv.putText(img,'%d'%i,vs[i] ,cv.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

    cv.line(img, vs[0], vs[1], cls[0], 2, 8, 0)
    cv.line(img, vs[1], vs[2], cls[1], 2, 8, 0)
    cv.line(img, vs[2], vs[3], cls[2], 2, 8, 0)
    cv.line(img, vs[3], vs[0], cls[3], 2, 8, 0)

    cv.line(img, vs[4], vs[5], cls[4], 2, 8, 0)
    cv.line(img, vs[5], vs[6], cls[5], 2, 8, 0)
    cv.line(img, vs[6], vs[7], cls[6], 2, 8, 0)
    cv.line(img, vs[7], vs[4], cls[7], 2, 8, 0)

    cv.line(img, vs[0], vs[4], cls[8], 2, 8, 0)
    cv.line(img, vs[3], vs[7], cls[9], 2, 8, 0)
    cv.line(img, vs[2], vs[6], cls[10], 2, 8, 0)
    cv.line(img, vs[1], vs[5], cls[11], 2, 8, 0)


def draw_3d_box2(img,bvs,stx,sty):
    col1 = (0,0,255)
    col2 = (255,0,0)

    cls = [col1, col1, col1, col1, col2, col2, col1, col1, col1, col1, col1, col2]

    vs = []
    for var in bvs:
        vs.append((int(stx+var[0]),int(sty+var[1])))

    #for i in range(8):
    #    cv.putText(img,'%d'%i,vs[i] ,cv.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

    cv.line(img, vs[0], vs[1], cls[0], 2, 8, 0)
    cv.line(img, vs[1], vs[2], cls[1], 2, 8, 0)
    cv.line(img, vs[2], vs[3], cls[2], 2, 8, 0)
    cv.line(img, vs[3], vs[0], cls[3], 2, 8, 0)

    cv.line(img, vs[4], vs[5], cls[4], 2, 8, 0)
    cv.line(img, vs[5], vs[6], cls[5], 2, 8, 0)
    cv.line(img, vs[6], vs[7], cls[6], 2, 8, 0)
    cv.line(img, vs[7], vs[4], cls[7], 2, 8, 0)

    cv.line(img, vs[0], vs[4], cls[8], 2, 8, 0)
    cv.line(img, vs[3], vs[7], cls[9], 2, 8, 0)
    cv.line(img, vs[2], vs[6], cls[10], 2, 8, 0)
    cv.line(img, vs[1], vs[5], cls[11], 2, 8, 0)

def play_with_box2():
    def nothing(x):
        pass

    w, h = 300, 200

    cv.namedWindow('video')
    cv.createTrackbar('rs', 'video', 0, 100, nothing)
    cv.createTrackbar('theta1', 'video', 0, 90, nothing)
    cv.createTrackbar('theta2', 'video', 0, 90, nothing)
    while True:
        img = np.ones([500, 500, 3], np.uint8) * 255

        cv.rectangle(img, (100, 100), (100 + w, 100 + h), (0, 255, 0), 2)
        rs = cv.getTrackbarPos('rs', 'video')
        theta1 = cv.getTrackbarPos('theta1', 'video')
        theta2 = cv.getTrackbarPos('theta2', 'video')

        rs = rs / 100.0
        theta1 = (np.pi/2.0) * (theta1/90.0)
        theta2 = (np.pi / 2.0) * (theta2 / 90.0)

        box_vs = compute_box_2(rs,theta1,theta2,w,h)
        draw_3d_box2(img,box_vs,100,100)
        cv.imshow('video', img)
        cv.waitKey(50)
        del (img)




def play_with_box():
    def nothing(x):
        pass

    w, h = 300, 200

    cv.namedWindow('video')
    cv.createTrackbar('rs0', 'video', 0, 100, nothing)
    cv.createTrackbar('rs1', 'video', 0, 100, nothing)
    cv.createTrackbar('rs2', 'video', 0, 100, nothing)

    while True:
        img = np.ones([700, 1000, 3], np.uint8) * 255

        cv.rectangle(img, (100, 100), (100 + w, 100 + h), (0, 255, 0), 2)
        cv.rectangle(img, (600, 100), (600 + w, 100 + h), (0, 255, 0), 2)
        cv.rectangle(img, (100, 400), (100 + w, 400 + h), (0, 255, 0), 2)
        cv.rectangle(img, (600, 400), (600 + w, 400 + h), (0, 255, 0), 2)

        rs0 = cv.getTrackbarPos('rs0', 'video')
        rs1 = cv.getTrackbarPos('rs1', 'video')
        rs2 = cv.getTrackbarPos('rs2','video')


        if rs1 < rs2:
            rs2 = rs1

        box_vs1,box_vs2 = compute_box(w,h,rs0,rs1,rs2)

        draw_3d_box(img,box_vs1,100,100,0)
        draw_3d_box(img,box_vs2,600,100,0)
        draw_3d_box(img, box_vs1, 100, 400, 1)
        draw_3d_box(img, box_vs2, 600, 400, 1)

        draw_arrow(img,box_vs1,100,100,w,h,0)
        draw_arrow(img,box_vs2,600,100,w,h,1)
        draw_arrow(img,box_vs1,100,400,w,h,2)
        draw_arrow(img,box_vs2,600,400,w,h,3)

        cv.imshow('video', img)
        cv.waitKey(50)
        del(img)



if __name__ == '__main__':
    play_with_box2()









































