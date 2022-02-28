import cv2
import numpy as np
import os
from pywt import dwt2, idwt2


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def cv_imwrite(path, img):
    suffix = os.path.splitext(path)[-1]
    cv2.imencode(suffix, img)[1].tofile(path)


class WaterMark():
    def __init__(self, RandomSeedWM, RandomSeedDCT, WMShape=None):
        self.BlockShape = (4, 4)  # 分块大小4*4
        self.RandomSeedWM = RandomSeedWM  # 水印置乱随机数种子
        self.RandomSeedDCT = RandomSeedDCT  # DCT变换随机数种子
        # Mod, Mod2影响水印鲁棒性和图片失真性, 为插入水印的参数
        self.Mod = 36
        self.Mod2 = 20
        self.WMShape = WMShape  # 水印的大小, 提取水印时应先知道水印的大小

    def InitBlockAddIndex(self, ImgShape):  # 判断水印是否超过图片大小, 并进行一些初始化
        # 计算图像分为多少块
        Shape0Int, Shape1Int = ImgShape[0] // self.BlockShape[0], ImgShape[1] // self.BlockShape[1]
        if Shape0Int * Shape1Int < self.WMShape[0] * self.WMShape[1]:
            print('水印的大小超过图片的容量, 建议更改')
        self.PartShape = (
            Shape0Int * self.BlockShape[0], Shape1Int * self.BlockShape[1])  # 计算图片被分块的区域
        self.BlockAddIndex0, self.BlockAddIndex1 = np.meshgrid(
            np.arange(Shape0Int), np.arange(Shape1Int))
        self.BlockAddIndex0, self.BlockAddIndex1 = self.BlockAddIndex0.flatten(
        ), self.BlockAddIndex1.flatten()  # 生成图片被分块的坐标
        self.Length = self.BlockAddIndex0.size  # 生成图片被分块的个数

    def ReadOriImg(self, FileName):  # 读取图片
        OriImg = cv2.imread(FileName).astype(np.float32)  # 读取图片
        self.OriImgShape = OriImg.shape[:2]  # 取图片大小
        self.OriImgYUV = cv2.cvtColor(
            OriImg, cv2.COLOR_BGR2YUV)  # 图片RGB格式转为YUV格式

        # 预设图片一级DWT变换
        # 判断图片能否平分为二, 若不能则填充
        if self.OriImgYUV.shape[0] % 2 != 0:  # 宽度填充
            self.OriImgYUV = np.concatenate(
                (self.OriImgYUV, np.zeros((1, self.OriImgYUV.shape[1], 3))), axis=0)
        if self.OriImgYUV.shape[1] % 2 != 0:  # 高度填充
            self.OriImgYUV = np.concatenate(
                (self.OriImgYUV, np.zeros((self.OriImgYUV.shape[0], 1, 3))), axis=1)

        # 判断是否满足上述条件
        assert self.OriImgYUV.shape[0] % 2 == 0
        assert self.OriImgYUV.shape[1] % 2 == 0

        CoeffsY, CoeffsU, CoeffsV = dwt2(self.OriImgYUV[:, :, 0], 'haar'), dwt2(
            self.OriImgYUV[:, :, 1], 'haar'), dwt2(self.OriImgYUV[:, :, 2], 'haar')  # 对原始图像做DWT变换
        HaY, HaU, HaV = CoeffsY[0], CoeffsU[0], CoeffsV[0]  # 提取低频部分

        # DWT结果加入类属性
        self.CoeffsY, self.CoeffsU, self.CoeffsV = CoeffsY[1], CoeffsU[1], CoeffsV[1]
        self.HaY, self.HaU, self.HaV = HaY, HaU, HaV  # DWT结果加入类属性
        self.HaBlockShape = (
            self.HaY.shape[0] // self.BlockShape[0], self.HaY.shape[1] // self.BlockShape[1], self.BlockShape[0], self.BlockShape[1])  # 将低频图片分块, 分成若干个4*4的小块
        Strides = self.HaY.itemsize * \
            (np.array([self.HaY.shape[1] * self.BlockShape[0],
             self.BlockShape[1], self.HaY.shape[1], 1]))

        # 将低频图片分块, 分成若干个4*4的小块
        self.HaYBlock = np.lib.stride_tricks.as_strided(
            self.HaY.copy(), self.HaBlockShape, Strides)
        self.HaUBlock = np.lib.stride_tricks.as_strided(
            self.HaU.copy(), self.HaBlockShape, Strides)
        self.HaVBlock = np.lib.stride_tricks.as_strided(
            self.HaV.copy(), self.HaBlockShape, Strides)

    def ReadWM(self, FileName):  # 读取水印
        self.WM = cv2.imread(FileName)[:, :, 0]  # 提取图片B通道
        self.WMShape = self.WM.shape[:2]  # 提取水印大小
        self.InitBlockAddIndex(self.HaY.shape)  # 初始化块索引数组, 并判断块是否足够存储水印信息
        self.WMFlatten = self.WM.flatten()  # 水印变为一维
        self.RandomWM = np.random.RandomState(self.RandomSeedWM)  # 生成置乱水印随机化种子
        self.RandomWM.shuffle(self.WMFlatten)  # 置乱水印

    def BlockAddWM(self, block, index, i):
        # 计算i索引水印图片的位置, 因为图片大小大于水印图片, 取余
        i = i % (self.WMShape[0] * self.WMShape[1])

        wm_1 = self.WMFlatten[i]  # 取水印图片相应的位置
        BlockDCT = cv2.dct(block)  # 计算DCT
        BlockDCTFlatten = BlockDCT.flatten().copy()  # DCT结果变为一维
        BlockDCTFlatten = BlockDCTFlatten[index]  # DCT结果置乱
        BlockDCTShuffled = BlockDCTFlatten.reshape(self.BlockShape)  # 置乱后变为二维
        U, s, V = np.linalg.svd(BlockDCTShuffled)  # 对DCT置乱后做SVD分解取特征值

        # 将水印信息的灰阶影响嵌入到奇异值矩阵
        max_s = s[0]
        s[0] = (max_s - max_s % self.Mod + 3 / 4 *
                self.Mod) if wm_1 >= 128 else (max_s - max_s % self.Mod + 1 / 4 * self.Mod)
        max_s = s[1]
        s[1] = (max_s - max_s % self.Mod2 + 3 / 4 *
                self.Mod2) if wm_1 >= 128 else (max_s - max_s % self.Mod2 + 1 / 4 * self.Mod2)

        BlockDCTShuffled = np.dot(U, np.dot(np.diag(s), V))  # SVD分解逆过程
        BlockDCTFlatten = BlockDCTShuffled.flatten()  # 变一维
        BlockDCTFlatten[index] = BlockDCTFlatten.copy()  # 置乱逆操作
        BlockDCT = BlockDCTFlatten.reshape(self.BlockShape)  # 变二维
        return cv2.idct(BlockDCT)  # DCT逆变换

    def Embed(self, FileName):  # 嵌入水印
        EmbedHaYBlock, EmbedHaUBlock, EmbedHaVBlock = self.HaYBlock.copy(
        ), self.HaUBlock.copy(), self.HaVBlock.copy()  # 复制, 防止引用改变类属性

        self.RandomDCT = np.random.RandomState(
            self.RandomSeedDCT)  # 生成DCT变换随机种子
        index = np.arange(self.BlockShape[0] * self.BlockShape[1])  # 生成块像素索引

        for i in range(self.Length):  # 遍历所有块
            self.RandomDCT.shuffle(index)  # 随机打乱索引
            # 将水印嵌入图像的DWT变换的低频域
            EmbedHaYBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]] = self.BlockAddWM(
                EmbedHaYBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index, i)
            EmbedHaUBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]] = self.BlockAddWM(
                EmbedHaUBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index, i)
            EmbedHaVBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]] = self.BlockAddWM(
                EmbedHaVBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index, i)

        # 合并分块
        EmbedHaYPart = np.concatenate(EmbedHaYBlock, 1)
        EmbedHaYPart = np.concatenate(EmbedHaYPart, 1)
        EmbedHaUPart = np.concatenate(EmbedHaUBlock, 1)
        EmbedHaUPart = np.concatenate(EmbedHaUPart, 1)
        EmbedHaVPart = np.concatenate(EmbedHaVBlock, 1)
        EmbedHaVPart = np.concatenate(EmbedHaVPart, 1)

        # 补回未被分块嵌入的部分
        EmbedHaY = self.HaY.copy()
        EmbedHaY[:self.PartShape[0], :self.PartShape[1]] = EmbedHaYPart
        EmbedHaU = self.HaU.copy()
        EmbedHaU[:self.PartShape[0], :self.PartShape[1]] = EmbedHaUPart
        EmbedHaV = self.HaV.copy()
        EmbedHaV[:self.PartShape[0], :self.PartShape[1]] = EmbedHaVPart

        # DWT逆变换
        EmbedHaY = idwt2((EmbedHaY.copy(), self.CoeffsY),
                         'haar')  # 其idwt得到父级的ha
        EmbedHaU = idwt2((EmbedHaU.copy(), self.CoeffsU),
                         'haar')  # 其idwt得到父级的ha
        EmbedHaV = idwt2((EmbedHaV.copy(), self.CoeffsV),
                         'haar')  # 其idwt得到父级的ha

        # 还原图像
        EmbedImgYUV = np.zeros(self.OriImgYUV.shape, dtype=np.float32)
        EmbedImgYUV[:, :, 0] = EmbedHaY
        EmbedImgYUV[:, :, 1] = EmbedHaU
        EmbedImgYUV[:, :, 2] = EmbedHaV
        # 去掉最初填充的部分
        EmbedImgYUV = EmbedImgYUV[:self.OriImgShape[0], :self.OriImgShape[1]]
        EmbedImg = cv2.cvtColor(EmbedImgYUV, cv2.COLOR_YUV2BGR)  # YUV变为RGB

        # 规范超出范围的部分
        EmbedImg[EmbedImg > 255] = 255
        EmbedImg[EmbedImg < 0] = 0

        cv2.imwrite(FileName, EmbedImg)

    def BlockGetWM(self, block, index):  # 提取水印
        BlockDCT = cv2.dct(block)
        BlockDCTFlatten = BlockDCT.flatten().copy()
        BlockDCTFlatten = BlockDCTFlatten[index]
        BlockDCTShuffled = BlockDCTFlatten.reshape(self.BlockShape)
        U, s, V = np.linalg.svd(BlockDCTShuffled)
        max_s = s[0]
        wm_1 = 255 if max_s % self.Mod > self.Mod / 2 else 0
        max_s = s[1]
        wm_2 = 255 if max_s % self.Mod2 > self.Mod2 / 2 else 0
        wm = (wm_1 * 3 + wm_2 * 1) / 4
        return wm

    def Extract(self, FileName, OutWMName):
        if not self.WMShape:
            print('水印形状未设定')
            return 0

        EmbedImg = cv2.imread(FileName).astype(np.float32)
        EmbedImgYUV = cv2.cvtColor(EmbedImg, cv2.COLOR_BGR2YUV)

        if EmbedImgYUV.shape[0] % 2 != 0:  # 宽度填充
            EmbedImgYUV = np.concatenate(
                (EmbedImgYUV, np.zeros((1, EmbedImgYUV.shape[1], 3))), axis=0)
        if EmbedImgYUV.shape[1] % 2 != 0:  # 高度填充
            EmbedImgYUV = np.concatenate(
                (EmbedImgYUV, np.zeros((EmbedImgYUV.shape[0], 1, 3))), axis=1)

        assert EmbedImgYUV.shape[0] % 2 == 0
        assert EmbedImgYUV.shape[1] % 2 == 0

        EmbedImgY, EmbedImgU, EmbedImgV = EmbedImgYUV[:,
                                                      :, 0], EmbedImgYUV[:, :, 1], EmbedImgYUV[:, :, 2]
        CoeffsY, CoeffsU, CoeffsV = dwt2(EmbedImgY, 'haar'), dwt2(
            EmbedImgU, 'haar'), dwt2(EmbedImgV, 'haar')
        HaY, HaU, HaV = CoeffsY[0], CoeffsU[0], CoeffsV[0]
        self.InitBlockAddIndex(HaY.shape)

        HaBlockShape = (HaY.shape[0] // self.BlockShape[0], HaY.shape[1] //
                        self.BlockShape[1], self.BlockShape[0], self.BlockShape[1])
        Strides = HaY.itemsize * \
            (np.array([HaY.shape[1] * self.BlockShape[0],
             self.BlockShape[1], HaY.shape[1], 1]))

        HaYBlock = np.lib.stride_tricks.as_strided(
            HaY.copy(), HaBlockShape, Strides)
        HaUBlock = np.lib.stride_tricks.as_strided(
            HaU.copy(), HaBlockShape, Strides)
        HaVBlock = np.lib.stride_tricks.as_strided(
            HaV.copy(), HaBlockShape, Strides)

        ExtractWM = np.array([])

        self.RandomDCT = np.random.RandomState(self.RandomSeedDCT)

        index = np.arange(self.BlockShape[0] * self.BlockShape[1])
        for i in range(self.Length):
            self.RandomDCT.shuffle(index),
            WMY = self.BlockGetWM(
                HaYBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index)
            WMU = self.BlockGetWM(
                HaUBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index)
            WMV = self.BlockGetWM(
                HaVBlock[self.BlockAddIndex0[i], self.BlockAddIndex1[i]], index)
            WM = round((WMY + WMU + WMV) / 3)

            if i < self.WMShape[0] * self.WMShape[1]:
                ExtractWM = np.append(ExtractWM, WM)
            else:
                times = i // (self.WMShape[0] * self.WMShape[1])
                ii = i % (self.WMShape[0] * self.WMShape[1])
                ExtractWM[ii] = (ExtractWM[ii] * times + WM) / (times + 1)

        WMIndex = np.arange(ExtractWM.size)
        self.RandomWM = np.random.RandomState(self.RandomSeedWM)
        self.RandomWM.shuffle(WMIndex)
        ExtractWM[WMIndex] = ExtractWM.copy()

        cv2.imwrite(OutWMName, ExtractWM.reshape(
            self.WMShape[0], self.WMShape[1]))


if __name__ == '__main__':
    BWM = WaterMark(4399, 2333)
    BWM.ReadOriImg('./test3.png')
    BWM.ReadWM('./WaterMark1.png')
    BWM.Embed('./out.png')

    BWM1 = WaterMark(4399, 2333, WMShape=(100, 100))
    BWM1.Extract('./out.png', './out_wm.png')
