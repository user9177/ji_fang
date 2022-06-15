# coding: utf-8
import cv2
import math
import numpy as np


def line():
    def nothing(x):
        pass

    img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    img_sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    img_sobelxy = cv2.addWeighted(cv2.convertScaleAbs(img_sobelx), 0.5, cv2.convertScaleAbs(img_sobely), 0.5, 0)

    edges = cv2.Canny(img, 150, 250, apertureSize=3)
    cv2.namedWindow('res')
    cv2.createTrackbar('distance', 'res', 1, 10, nothing)
    cv2.createTrackbar('threshold', 'res', 0, 250, nothing)
    cv2.createTrackbar('length', 'res', 1, 100, nothing)
    cv2.createTrackbar('gap', 'res', 1, 100, nothing)

    while True:
        height = 512
        width = 512
        new_img = np.zeros((height, width, 3), np.uint8)
        new_img[:, :] = (255, 0, 0)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        distance = cv2.getTrackbarPos('distance', 'res')
        threshold = cv2.getTrackbarPos('threshold', 'res')
        length = cv2.getTrackbarPos('length', 'res')
        gap = cv2.getTrackbarPos('gap', 'res')
        lines = cv2.HoughLinesP(edges, distance, np.pi / 180, threshold=threshold, minLineLength=length, maxLineGap=gap)
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(new_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('res', new_img)
    # distance = 1
    # threshold = 70
    # length = 20
    # gap = 27


# 矢量转换, 如果是负向量，转为正向量
def convert_vector(line):
    if line[0][0] > line[1][0]:
        line[0], line[1] = line[1], line[0]
    if line[0][0] == line[1][0] and line[0][1] > line[1][1]:
        line[0], line[1] = line[1], line[0]
    return line


def calc_angle(lineA, lineB):
    line1Y1 = lineA[0][1]
    line1X1 = lineA[0][0]
    line1Y2 = lineA[1][1]
    line1X2 = lineA[1][0]

    line2Y1 = lineB[0][1]
    line2X1 = lineB[0][0]
    line2Y2 = lineB[1][1]
    line2X2 = lineB[1][0]

    angle1 = math.atan2(line1Y1 - line1Y2, line1X1 - line1X2)
    angle2 = math.atan2(line2Y1 - line2Y2, line2X1 - line2X2)
    angleDegrees = (angle1 - angle2) * 360 / (2 * math.pi)
    return angleDegrees


# 计算线段的长度，用来剔除掉一些比较短的线条
def calc_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def pointer_at(point_img):
    edges = cv2.Canny(point_img, 150, 250, apertureSize=3)
    distance = 1
    threshold = 70
    length = 20
    gap = 27
    pointers = []  # 指针
    lines = cv2.HoughLinesP(edges, distance, np.pi / 180, threshold=threshold, minLineLength=length, maxLineGap=gap)
    ensure_edge = None
    ensure_edge_len = 0
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            line_length = calc_length(x1, y1, x2, y2)
            if line_length > ensure_edge_len:
                ensure_edge_len = line_length
                ensure_edge = [(x1, y1), (x2, y2)]
            # if 160 < line_length < 200:  # 太短或者太长的线条，不太可能是指针，就不要了
            pointers.append([(x1, y1), (x2, y2)])

    real_angle = 0
    for p in pointers:
        angle = calc_angle(convert_vector(p), convert_vector(ensure_edge))
        if 2 < angle < 88:
            real_angle = angle
        if angle < 0:
            if 2 < 270 + angle < 88:
                real_angle = 270 + angle
    out = real_angle / 90 * 12
    return print('%.2f' % out, 'KV')


if __name__ == "__main__":
    path = '1.jpg'
    point_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pointer_at(point_img)
