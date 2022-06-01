from email.mime import base
import cv2
from cv2 import circle
import numpy as np
import matplotlib.pyplot as plt
import datetime

def getCurrentTime():
    time = datetime.datetime.now()
    return datetime.datetime.strftime(time, '%Y-%m-%d %H:%M:%S')

def findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
    '''
    cv2.RETR_EXTERNAL     表示只检测外轮廓
    cv2.RETR_LIST           检测的轮廓不建立等级关系
    cv2.RETR_CCOMP          建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE            建立一个等级树结构的轮廓。

    第三个参数method为轮廓的近似办法
    cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法

    :param image:
    :return:  contours, hierarchy
    '''
    contours, hierarchy = cv2.findContours(image, mode, method)
    return contours, hierarchy

def drawContours(image, contours, color=(0,0,255)):
    for it in contours:
        cv2.drawContours(image, it, -1, color, 3)
    return image

def findMinRect(binary_img, thickness=2):
    '''
    CV_RETR_EXTERNAL：只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略；

    CV_RETR_LIST：检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓，所以hierarchy向量内所有元素的第3、第4个分量都会被置为-1，具体下文会讲到；

    CV_RETR_CCOMP: 检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层；

    CV_RETR_TREE: 检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。

    参数5：定义轮廓的近似方法，取值如下：
    :param binary_img:
    :return: [[x, y, w, h], ], img, rectImg

    w/h > 7 or w/h < 1/7  -> img
    '''
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    areaRectList1 = []
    areaRectList2 = []

    img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    for c in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        if w/h > 7 or w/h < 1/7:
            binary_img[y:y+h, x:x+w] = 0
            continue
        areaRectList1.append([x, y, w, h])
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), thickness)


        # # 找面积最小的矩形
        # rect = cv2.minAreaRect(c)
        # # 得到最小矩形的坐标
        # box = cv2.boxPoints(rect)
        # # 标准化坐标到整数
        # box = np.int0(box)
        # areaRectList2.append([box])
        # 画出边界
        #cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
    return areaRectList1, binary_img, img_rgb

#
# import cv2
# import numpy as np
#
# image = cv2.imread('new.jpg')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in contours:
#     # 找到边界坐标
#     x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     # 找面积最小的矩形
#     rect = cv2.minAreaRect(c)
#     # 得到最小矩形的坐标
#     box = cv2.boxPoints(rect)
#     # 标准化坐标到整数
#     box = np.int0(box)
#     # 画出边界
#     cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
#     # 计算最小封闭圆的中心和半径
#     (x, y), radius = cv2.minEnclosingCircle(c)
#     # 换成整数integer
#     center = (int(x),int(y))
#     radius = int(radius)
#     # 画圆
#     cv2.circle(image, center, radius, (0, 255, 0), 2)
#
# cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
# cv2.imshow("img", image)
# cv2.imwrite("img_1.jpg", image)
# cv2.waitKey(0)


def plot_show(fig, axes, plt_shows, savePath=None):
    '''
    ax2 = ax1.twinx()         # 让2个子图的x轴一样，同时创建副坐标轴。
    ax1.plot(x1, y1)
    :param fig:
    :param axes:
    :return:
    '''
    # 定义fig
    # # Create figure with sub-plots.
    # fig, axes = plt.subplots(3, 3)
    # assert len(plt_shows) < 9
    #
    # # Adjust vertical spacing if we need to print ensemble and best-net.
    # fig.subplots_adjust(hspace=0., wspace=0.3)

    if False:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate(axes.flat):
        # Plot image
        if len(plt_shows) <= i:
            continue
        ax.imshow(plt_shows[i].image, cmap="gray", interpolation=interpolation)

        xlabel = plt_shows[i].name
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    if savePath is not None:
        plt.savefig(savePath)
    plt.show()

def eroded(image, iter=1):
    '''
    腐蚀图像
    :param image:
    :return:
    '''
    # 椭圆：cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 矩形：cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # OpenCV定义的结构矩形元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(image, kernel, iterations=iter)  # 腐蚀图像
    return eroded

def dilated(image, iter=1):
    '''
    膨胀图像
    :param image:
    :return:
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(image, kernel, iterations=iter)  # 膨胀图像

    return dilated

def morphClose(image, size=(3,3), iter=1):
    '''
    闭运算1
    :param image:
    :return:
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 定义矩形结构元素
    closed1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iter)  # 闭运算1
    return closed1

def morphOpen(image, size=(3,3), iter=1):
    '''
    开运算1
    :param image:
    :return:
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 定义矩形结构元素
    opened1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=iter)     #开运算1
    return  opened1

def morphGrad(image):
    '''
    梯度
    :param image:
    :return:
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)             #梯度
    return gradient



def avgBlur(image, ksize=5):
    '''
    均值滤波(计算简单，速度快，但容易使照片模糊)
    :param image:
    :param ksize:
    :return:
    '''
    return cv2.blur(image,(ksize,ksize))

def gaussBlur(image, ksize=3):
    '''
    高斯滤波(中心点权重较高，周围点权重较低，比较适合处理高斯噪声);
    :param image:
    :param ksize:
    :return:
    '''
    return cv2.GaussianBlur(image, (ksize, ksize), 1)
#
def medianBlur(image, ksize=5):
    '''
    中值滤波(使用邻域灰度值的中值作为中心点灰度值,非常适合处理椒盐噪声)
    :param image:
    :param ksize:
    :return:
    '''
    return cv2.medianBlur(image,ksize)


def find_number_line(image, show=False):
    result = ''

    shape = image.shape
    assert len(shape) == 2
    h, w = shape

    line1_idx = int(w/2)
    line2_idx = int(h/4)
    line3_idx = int(3*h/4)

    p11 = (line1_idx, 0)
    p12 = (line1_idx, h)

    p21 = (0, line2_idx)
    p22 = (w, line2_idx)

    p31 = (0, line3_idx)
    p32 = (w, line3_idx)

    if show:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.line(img_bgr, p11, p12, (0,255,0))
        cv2.line(img_bgr, p21, p22, (0, 255, 0))
        cv2.line(img_bgr, p31, p32, (0, 255, 0))
        cv2.imshow("find_number_line", img_bgr)
        cv2.waitKey(0)

    def aa(line_img):
        template = '0123'
        init = [0, 0.12, 0.24, 0.36]
        line1_rate = np.mean(line_img/255)
        res = np.argmin(abs(init - line1_rate))
        return template[res]

    template = '0123'
    init = [0, 0.12, 0.24, 0.36]

    line_img = image[:, line1_idx]
    line_rate = np.mean(line_img / 255)
    res = np.argmin(abs(init - line_rate))
    result += str(res)

    init = [0, 0.21, 0.42, 0.63]
    line_img = image[line2_idx, :]
    line_rate = np.mean(line_img / 255)
    res = np.argmin(abs(init - line_rate))
    result += str(res)

    line_img = image[line3_idx, :]
    line_rate = np.mean(line_img / 255)
    res = np.argmin(abs(init - line_rate))
    result += str(res)
    return  result

def showImg(image, name="show-test"):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_circle_demo(image):
    print(image.shape)
    img = cv2.pyrMeanShiftFiltering(image, 10, 100)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=5, maxRadius=0)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        x, y, r = i[0], i[1], i[2]
        cv2.circle(image, (x, y), r, (0,255,0), 2)

    showImg(image=image)

    return image

def showTemplate_orgImg(orgImg, templateImg, thickness=1):
    shape = orgImg.shape
    w, h = templateImg.shape[0], templateImg.shape[1]

    if len(shape) == 2:
        orgImg = cv2.cvtColor(orgImg, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(orgImg, (0, 0), (h, w), (0, 0, 255), thickness=thickness)
    cv2.namedWindow("showTemplate_orgImg", 0)
    cv2.imshow("showTemplate_orgImg", orgImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



class PltShow():
    '''
    plt_shows = []
    plt_shows.append(PltShow(name="input name", image=image.copy()))
    ...
    pltSize = 4
    fig, axes = plt.subplots(pltSize, pltSize)
    assert len(plt_shows) <= pltSize*pltSize
    # Adjust vertical spacing if we need to print ensemble and best-net.
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    base.plot_show(fig=fig, axes=axes, plt_shows=plt_shows, savePath='./plt.jpg')
    '''
    def __init__(self, name, image):
        self.name = name
        self.image = image

print(getCurrentTime())
if __name__ == "__main__":

    pass
    
    path = "/home/ruitu/ljl/jifang/template_demo/target.jpg"
    img = cv2.imread(path)
    detect_circle_demo(img)