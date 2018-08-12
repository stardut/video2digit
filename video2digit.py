# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2

def img2digit(img):
    row, col = img.shape
    img = cv2.resize(img, (col // 2, row // 2))

    matrix = pooling(np.copy(img), 5)
    row, col = matrix.shape
    img_size = (int(row*14.8), int(col*12))
    
    img1 = np.ones(img_size, dtype=np.uint8) * 255
    res = text2img(matrix, np.copy(img1))
    row, col = res.shape
    img_t = cv2.resize(res, (int(col / 1), int(row / 1)))
    # cv2.imshow('test2', img_t)
    # cv2.waitKey(0)
    return img_t


def text2img(matrix, img):
    for rdx, rows in enumerate(matrix):        
        cv2.putText(img, ''.join(map(str, rows)), (0, rdx*15), 
            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
    return img

def pooling(img, pool_size=2):
    row, col = img.shape
    s_row = row // pool_size * pool_size
    s_col = col // pool_size * pool_size

    img = cv2.resize(img, (s_col, s_row))
    img = img // 25

    res = np.zeros((s_row // pool_size, s_col // pool_size), np.int32)

    for r in range(s_row // pool_size):
        for c in range(s_col // pool_size):
            res[r, c] = max_pooling(
                img[r * pool_size : (r+1) * pool_size , c * pool_size : (c+1) * pool_size],
                pool_size)
    return res


def max_pooling(img, pool_size):
    res = 0
    for r in range(pool_size):
        for c in range(pool_size):
           res = max(res, img[r, c])
    return res


if __name__ == '__main__':
    cap = cv2.VideoCapture("test.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = img2digit(img)

    y, x = res.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if os.path.exists('target.avi'):
        os.remove('target.avi')
        print('delete file')

    videoWriter = cv2.VideoWriter('target.avi', fourcc, fps, (x, y), False)
    num = 0
    while(1):
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = img2digit(img)
        videoWriter.write(res)
        print(num, end=', ')
        print(num//24)
        num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    videoWriter.release()    

