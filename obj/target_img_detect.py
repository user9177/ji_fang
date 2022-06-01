import cv2 as cv
import cv2
from utils import base


def deal_with(img_org):
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_org, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", binary)
    morph = base.morphOpen(binary, size=(11, 11))
    morph = base.morphClose(morph, size=(7, 7), iter=2)
    boxes_list, delRectImg, drawRectImg = base.findMinRect(morph, 5)
    if len(boxes_list) != 0:
        return True


def judge(picture_path, Device_ID):
    light_state_list = []
    targetImg = cv2.imread(picture_path)
    target_w = int(targetImg.shape[1])
    target_h = int(targetImg.shape[0])
    if Device_ID == "a":
        numbers = 3
    if Device_ID == "b":
        numbers = 2
    if Device_ID == "c":
        numbers = 8
    else:
        numbers = 10
    for i in range(numbers):
        x = int(target_w / numbers)
        # y = int(target_h/2)
        cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (0, 0, 255), 2)
        cv2.imshow('org', targetImg)
        cv2.waitKey(0)
        img_org = targetImg[0:int(target_h), i * x:(i + 1) * x]
        if deal_with(img_org):
            cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (255, 0, 0), 2)
            cv.putText(targetImg, "on", (i * x, 25), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
            light_state_list.append(["on", [i * x, 0, (i + 1) * x, int(target_h)]])
        else:
            cv2.rectangle(targetImg, (i * x, 0), ((i + 1) * x, int(target_h)), (255, 0, 0), 2)
            cv.putText(targetImg, "off", (i * x, 25), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
            light_state_list.append(["off", [i * x, 0, (i + 1) * x, int(target_h)]])
    # 图片缩放
    img_rgb = cv2.resize(targetImg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("img_rgb", img_rgb)
    cv2.imshow('targetImg', targetImg)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return light_state_list


if __name__ == '__main__':
    # pic = "2g.jpg"
    pic = "8w.jpg"
    light_state_list = judge(pic, Device_ID="c")
    print("light_state_list:{}".format(light_state_list))

    # pic_path = "3g.jpg"
    # for pic in os.listdir(pic_path):
    #     task5(pic)
