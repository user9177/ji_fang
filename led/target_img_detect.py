import cv2 as cv
import cv2


def morphClose(image, size=(3, 3), iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 定义矩形结构元素
    closed1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iter)  # 闭运算1
    return closed1


def morphOpen(image, size=(3, 3), iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 定义矩形结构元素
    opened1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iter)  # 开运算1
    return opened1


def findMinRect(binary_img, thickness=2):
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaRectList1 = []
    img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    for c in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        if w / h > 7 or w / h < 1 / 7:
            binary_img[y:y + h, x:x + w] = 0
            continue
        areaRectList1.append([x, y, w, h])
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), thickness)
    return areaRectList1, binary_img, img_rgb


def deal_with(img_org):
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_org, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", binary)
    morph = morphOpen(binary, size=(11, 11))
    morph = morphClose(morph, size=(7, 7), iter=2)
    boxes_list, delRectImg, drawRectImg = findMinRect(morph, 5)
    if len(boxes_list) != 0:
        return True


def judge(targetImg, lightSubType):
    light_state_list = []
    target_w = int(targetImg.shape[1])
    target_h = int(targetImg.shape[0])
    light_dict = {'2': 2, '3': 3, '4': 4, '8': 8, '10': 10}
    if lightSubType in light_dict.keys():
        numbers = light_dict[lightSubType]
    else:
        print("This lightSubType don't exist：", lightSubType)
        return None
    for i in range(numbers):
        x = int(target_w / numbers)
        # y = int(target_h/2)
        cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (0, 0, 255), 2)
        img_org = targetImg[0:int(target_h), i * x:(i + 1) * x]
        if deal_with(img_org):
            cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (255, 0, 0), 2)
            cv.putText(targetImg, "1", (i * x, 25), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
            light_state_list.append("1")
        else:
            cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (255, 0, 0), 2)
            cv.putText(targetImg, "0", (i * x, 25), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
            light_state_list.append("0")

    return print(light_state_list)


if __name__ == '__main__':
    path = '2_1.jpg'
    targetImg = cv2.imread(path)
    light_state_list = judge(targetImg, lightSubType='2')
