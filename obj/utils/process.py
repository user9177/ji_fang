
import os
import time
import matplotlib.pyplot as plt
import cv2
from utils import base
from utils.base import PltShow
import numpy as np

def match(img_org, img_temp, threshold=0.5, method=5, group=True):
    '''

    :param img_org:
    :param img_temp:
    :param threshold:
    :param method: TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5
    :return:
    '''
    shape = img_org.shape
    if len(shape) == 3:
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    if len(img_temp.shape) == 3:
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    h, w = img_temp.shape[:2]
    if h >= shape[0] or w >= shape[1]:
        print("template img size too much")
        assert False

    # 归一化平方差匹配
    res = cv2.matchTemplate(img_org, img_temp, method=method)

    # 返回res中值大于0.8的所有坐标
    # 返回坐标格式(col,row) 注意：是先col后row 一般是(row,col)!!!
    loc = np.where(res >= threshold)

    len2 = len(loc[0])
    if len2 == 0:
        return None, None

    # loc：标签/行号索引 常用作标签索引
    # iloc：行号索引
    # loc[::-1]：取从后向前（相反）的元素
    # *号表示可选参数
    boxes=[]
    outimg = []
    for pt in zip(*loc[::-1]):
        boxes.append([pt[0], pt[1], w, h] )
        img_test = img_org[pt[1]:pt[1] + h, pt[0]:pt[0] + w].copy()
        # cv2.imshow("*****img_Test:", img_test)
        # cv2.waitKey(1)
        outimg.append(img_org[pt[1]:pt[1] + h, pt[0]:pt[0] + w].copy() )

    if group:
        boxes = np.array(boxes)
        boxes, weights = cv2.groupRectangles(boxes.tolist(), 1, 0.8)

    return boxes, outimg


def regNumber(orgImg, templateImg, templateID, templateScale, init_line_val, matchThreshold=0.7, show=False):
    '''
    :param orgImg:
    :param templateImg:
    :param templateID:
    :param templateScale:
    :param init_line_val:
    :param show:
    :return: [templateID, (x, y, w, h)]
    '''
    templateImg = cv2.resize(templateImg, (0, 0), fx=templateScale, fy=templateScale)
  #  base.showTemplate_orgImg(orgImg, templateImg)

    boxes, outImgs = match(orgImg, templateImg, threshold=matchThreshold, method=5, group=True)  # 0.7:num8,  0.8:num0, 9:
    img_0_bgr = cv2.cvtColor(orgImg, cv2.COLOR_GRAY2RGB)
    shape = templateImg.shape
    cv2.rectangle(img_0_bgr, (0, 0), (shape[1], shape [0]), (0, 0, 255), 2)
    if show:
        cv2.imshow("org img", img_0_bgr)
        cv2.imshow("template", templateImg)
    outBoxes = []
    if boxes is None:
        print('%d template not match')
        boxes = []

    for box in boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_0 = orgImg[y:y + h, x:x + w].copy()
        cv2.rectangle(img_0_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if show:
            cv2.imshow("match result", img_0_bgr)
            cv2.waitKey(0)

        if templateID != '.':
            res = base.find_number_line(img_0)
            if init_line_val[templateID] != res:
                continue
        outBoxes.append([templateID, (x, y, w, h)])
    return outBoxes

def process(image, config=None):
    class Numbers():
        '''
        self.locals:  [[x, y, w, h], ]
        self.IDs: [1, 4, 0, ]
        '''
        def __init__(self):
            self.locals = []
            self.ids = []
    numbers = Numbers()
    ids = []
    locals = []
    init_line_val = {'0':'222', '1':'001', '2':'311', '3':'311', '4':'121', '5':'311', '6':'312', '7':'111', '8':'322', '9':'321'}
    # path = "/home/ljl/my_projects/template_demo/tt.png"
    # image = cv2.imread(path, 0)
    # 显示图片 参数：（窗口标识字符串，imread读入的图像）
    shape = image.shape
    if len(shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt_shows = []
    plt_shows.append(PltShow(name="input name", image=image.copy()))

    # dotPath = "/home/ljl/my_projects/template_demo/number_template/dot/1.png"
    # dotImg = cv2.imread(dotPath)
    #
    # rate = 1.5
    # tempImg = cv2.resize(dotImg, (0, 0), fx=rate, fy=rate)
    # cv2.imshow("tempImg", tempImg)
    # img2 = cv2.resize(img, (0, 0), fx=rate, fy=rate)
    # boxes, outImgs = process.match(image, tempImg, threshold=thresholds[idx], method=5, group=True)
    # return  None

    binary = cv2.adaptiveThreshold(image.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)  # 25, 10   11
    #res, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    plt_shows.append(PltShow(name="binary", image=binary.copy()))

    '''
    binary = base.morphClose(binary, iter=1)
   # binary = base.dilated(binary, iter=1)
    list1, delRectImg, imgbgr = base.findMinRect(binary)
    count = 0
    for it in list1:
        x, y, w, h = it
        img = image[y:y+h, x:x+w]
        cv2.imwrite("temp/"+str(count) + ".jpg", img)
        count += 1

    cv2.imshow("show", imgbgr)
    cv2.waitKey(0)
    return None
    '''

    # morph
    morph = base.morphClose(binary, size=(5, 5))
    plt_shows.append(PltShow(name="morph_close", image=morph.copy()))

    morph2 = 255 - morph.copy()
    plt_shows.append(PltShow(name="inv img", image=morph2.copy()))

    morph2 = base.morphClose(morph2, size=(15, 15))
    plt_shows.append(PltShow(name="morph_close_2", image=morph2.copy()))

    del_row_idx = []
    del_col_idx = []
    w, h = morph2.shape
    threshold = 0.7
    for i in range(h):
        if np.sum(morph2[i:i+1, :]/255) / w > threshold:
            del_row_idx.append(i)
    for j in range(w):
        if np.sum(morph2[:, j:j+1]/255) / h > threshold:
            del_col_idx.append(j)
    for i in del_row_idx:
        morph2[i:i + 1, :] = 0
    for i in del_col_idx:
        morph2[:, i:i+1] = 0


    list1, delRectImg, _ = base.findMinRect(morph2)
    plt_shows.append(PltShow(name="del rect img", image=delRectImg.copy()))

    delRectImg = base.morphClose(delRectImg, size=(15, 15), iter=2)
    plt_shows.append(PltShow(name="morph_close_3", image=delRectImg.copy()))

    list1, delRectImg, imgbgr = base.findMinRect(delRectImg)
    plt_shows.append(PltShow(name="draw rect img", image=imgbgr.copy()))

    #find much rate of w and h
    max_area = 0
    max_box = None
    for it in list1:
        x, y, w, h = it
        if w*h > max_area:
            max_area = w*h
            max_box = it

    if max_box is not None:
        x, y, w, h = max_box

        padsize = 10
        if x > padsize:
            x -= padsize
        if y > padsize:
            y -= padsize
        if (w+padsize+x) < image.shape[1]:
            w += padsize*2
        if (h+padsize+y) < image.shape[0]:
            h += padsize*2

        targetImg = image[y:y+h, x:x+w].copy()
        plt_shows.append(PltShow(name="target img", image=targetImg.copy()))
        cv2.imwrite("./result.jpg", targetImg)

        targetImg0 = cv2.adaptiveThreshold(targetImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)  # 25, 10
        plt_shows.append(PltShow(name="target brinary img", image=targetImg0.copy()))


        targetImg1 = cv2.GaussianBlur(targetImg, (3, 3), 1)
        plt_shows.append(PltShow(name="gauss img", image=targetImg1.copy()))

        # cv2.equalizeHist(targetImg1, targetImg1)
        # plt_shows.append(PltShow(name="gauss equalizeHist img", image=targetImg1.copy()))
        targetImg1 = cv2.adaptiveThreshold(255-targetImg1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)  # 25, 10
        plt_shows.append(PltShow(name="gauss brinary img", image=targetImg1.copy()))

        targetImg2 = base.morphOpen(targetImg1, size=(3, 3), iter=2)
        plt_shows.append(PltShow(name="gauss b morphO", image=targetImg2.copy()))
        cv2.imwrite('./test.jpg', targetImg2)

        number_temp = []
        number_id = []
        scals = [1.0, 0.9, 0.9, 0.5]
        number_temp.append("./obj1/number_template/9/9.jpg")
        number_temp.append("./obj1/number_template/0/0.png")
        number_temp.append("./obj1/number_template/8/8.png")
        number_temp.append("./obj1/number_template/dot/1.png")
        number_id.append('9')
        number_id.append('0')
        number_id.append('8')
        number_id.append('.')
        matchThreshold_list = [0.7,0.7,0.7,0.6]
        for i in range(len(number_temp)):
            # if number_id[i] != '.':
            #     continue
            assert os.path.exists(number_temp[i])
            number_img = cv2.imread(number_temp[i], 0)

            outs = regNumber(orgImg=targetImg2, templateImg=number_img, templateID=number_id[i],
                             templateScale=scals[i], init_line_val=init_line_val, matchThreshold=matchThreshold_list[i], show=False)
            for it in outs:
                ids.append(it[0])
                locals.append(it[1])


        ids_temp = []
        locals_temp = []

        dict_ = {}
        x_list = []
        for i in range(len(ids)):
            dict_[locals[i][0]] = i
            x_list.append(locals[i][0])
        x_list.sort()

        for it in x_list:
            ids_temp.append(ids[dict_[it]])
            locals_temp.append(locals[dict_[it]])

        ids = ids_temp
        locals = locals_temp


    resultImg = targetImg1.copy() * 0
    result = ''
    for it in ids:
        result += it
    cv2.putText(resultImg, result, (int(resultImg.shape[1]/2), int(resultImg.shape[0]/2)),cv2.FONT_HERSHEY_SIMPLEX,0.75 ,color=(255,255,255), thickness=2)
    plt_shows.append(PltShow(name="result", image=resultImg))

    pltSize = 4
    fig, axes = plt.subplots(pltSize, pltSize)
    assert len(plt_shows) <= pltSize*pltSize
    # Adjust vertical spacing if we need to print ensemble and best-net.
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    base.plot_show(fig=fig, axes=axes, plt_shows=plt_shows, savePath='./plt.jpg')

    return ids, locals

    # # 参数：
    # # 第一个参数
    # # 处理的原图像，该图像必须为单通道的灰度图；
    # # 第二个参数
    # # 最小阈值；
    # # 第三个参数
    # # 最大阈值。
    # img = cv2.GaussianBlur(binary, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
    # canny = cv2.Canny(img, 20, 150)  # 50是最小阈值,150是最大阈值
    # sobelImg = sobel.sobel(binary)
    # plt_shows.append(PltShow(name="canny img", image=canny.copy()))
    # 定义fig
    # Create figure with sub-plots.

if __name__ == "__main__":
    path = "/examples/saveAndLoadModel/mnist_save_load/00/8.png"
    image = cv2.imread(path, 0)
    binary = process(image)





#
# def test(path_org, path_temp, threshold=0.5):
#     # 1. 读入原图和模板
#     img_rgb = cv2.imread(path_org)
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#     template = cv2.imread(path_temp, 0)
#     h, w = template.shape[:2]
#
#     # 归一化平方差匹配
#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#
#
#     # 返回res中值大于0.8的所有坐标
#     # 返回坐标格式(col,row) 注意：是先col后row 一般是(row,col)!!!
#     loc = np.where(res >= threshold)
#
#     # loc：标签/行号索引 常用作标签索引
#     # iloc：行号索引
#     # loc[::-1]：取从后向前（相反）的元素
#     # *号表示可选参数
#     for pt in zip(*loc[::-1]):
#         right_bottom = (pt[0] + w, pt[1] + h)
#         print(pt)
#         cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
#
#     # 保存处理后的图片
#     cv2.imwrite('res.png', img_rgb)
#
#     # 显示图片 参数：（窗口标识字符串，imread读入的图像）
#     cv2.imshow("test_image", img_rgb)
#
#     # 窗口等待任意键盘按键输入 0为一直等待 其他数字为毫秒数
#     cv2.waitKey(0)
#
#     # 销毁窗口 退出程序
#     cv2.destroyAllWindows()
#     return  None, None
#





    #
    # row_list=[]
    # col_list=[]
    # del_threshold = 0.65
    # for i in range(morph.shape[0]):
    #     row_img = morph[i:i+1, :]
    #     aa = np.mean(row_img/255)
    #     if aa > del_threshold:
    #         row_list.append(i)
    # for j in range(morph.shape[1]):
    #     col_img = morph[j:j+1, :]
    #     aa = np.mean(col_img/255)
    #     if aa > del_threshold:
    #         col_list.append(j)
    #
    # for idx in row_list:
    #     morph[idx:idx+1, :] = 0
    # for idx in col_list:
    #     morph[:, idx:idx+1] = 0
    # plt_shows.append(PltShow(name="morph_close_2", image=morph.copy()))