from application.BlindWaterMark import *
import numpy as np
import cv2
import random


def NCC(A, B):
    cross_mul_sum = ((A-A.mean())*(B-B.mean())).sum()
    cross_square_sum = np.sqrt(
        (np.square(A-(A.mean())).sum())*(np.square(B-(B.mean())).sum()))
    return cross_mul_sum/cross_square_sum


def PSNR(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def test_ncc(filename1, filename2):
    a = cv_imread(filename1)
    b = cv_imread(filename2)
    return NCC(a, b)


def test_psnr(filename1, filename2):
    a = cv_imread(filename1)
    b = cv_imread(filename2)
    return PSNR(a, b)


def CutImgX(img, percent):  # 横向裁剪为原来的百分比
    out = np.zeros(img.shape, np.uint8)
    out[:int(512 * percent)] = img[:int(512 * percent)]
    return out


def CutImgY(img, percent):  # 纵向裁剪为原来的百分比
    out = np.zeros(img.shape, np.uint8)
    out[:, :int(512 * percent)] = img[:, :int(512 * percent)]
    return out


def RotateImg(img, anchor):  # 旋转
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, anchor, 1.0)
    out = cv2.warpAffine(img, M, (w, h))
    return out


def ResizeImg(img, percent):  # 缩放为原来的百分比
    out = np.zeros(img.shape, np.uint8)
    (h, w) = img.shape[:2]
    out[:int(h * percent), :int(w * percent)] = cv2.resize(img,
                                                           (int(h * percent), int(w * percent)))
    return out


def sp_noise(image, prob):  # 椒盐噪声
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return output


if __name__ == '__main__':
    img = cv2.imread('./AttrackTest/Embedded_Image0.png')
    cv2.imwrite(f'./AttrackTest/Embedded_Image{1}.png', CutImgX(img, 0.9))
    cv2.imwrite(f'./AttrackTest/Embedded_Image{2}.png', CutImgY(img, 0.9))
    cv2.imwrite(f'./AttrackTest/Embedded_Image{3}.png', RotateImg(img, 10))
    cv2.imwrite(f'./AttrackTest/Embedded_Image{4}.png', ResizeImg(img, 0.9))
    cv2.imwrite(f'./AttrackTest/Embedded_Image{5}.png', sp_noise(img, 0.005))

    for i in range(6):
        bwm = WaterMark(2333, 6666, (64, 64))
        bwm.Extract(
            f'./AttrackTest/Embedded_Image{i}.png', f'./AttrackTest/Extract{i}.png')

    for i in range(6):
        print(test_ncc('./AttrackTest/WaterMark.png',
              f'./AttrackTest/Extract{i}.png'))

