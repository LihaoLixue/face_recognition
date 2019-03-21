import dlib
import cv2
import os
# 导入cnn模型
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')  # 调用训练好的cnn进行人脸检测

img = cv2.imread(os.path.join(os.path.dirname(__file__), 'test_images', '003.jpg'))  # opencv 读取图片，并显示
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 取灰度

rects = cnn_face_detector(img, 1)  # 进行检测
print("Number of faces detected: {}".format(len(rects)))  # 打印检测到的人脸数

# 遍历返回的结果
# 返回的结果是一个mmod_rectangles对象。这个对象包含有2个成员变量：dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度。
for i, d in enumerate(rects):
    face = d.rect
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, face.left(), face.top(),
                                                                                      face.right(), d.rect.bottom(),
                                                                                      d.confidence))

    # cv2.rectangle()画出矩形,参数1：图像，参数2：矩形左上角坐标，参数3：矩形右下角坐标，参数4：画线对应的rgb颜色，参数5：线的宽度
    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

cv2.namedWindow("img", 2)  # #图片窗口可调节大小
cv2.imshow("img", img)  # 显示图像
cv2.waitKey(0)  # 等待按键，然后退出
