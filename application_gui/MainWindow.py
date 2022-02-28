from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showerror, showinfo

from BlindWaterMark import WaterMark
# from BlindWaterMark import WaterMark


class GUI(Tk):
    def __init__(self, title, geometry):  # 初始化窗口
        self.window = Tk()
        self.window.title(title)  # 窗口标题
        self.window.geometry(geometry)  # 窗口默认大小

        self.OriImgPath = StringVar()  # 原始图片文件路径, 作为文件打开
        self.WMPath = StringVar()  # 水印图片文件路径, 作为文件打开
        self.OutImgPath = StringVar()  # 嵌入水印的图片的文件路径, 作为文件打开

        self.RandomSeedWM = StringVar()  # 水印加密种子
        self.RandomSeedDCT = StringVar()  # DCT加密种子

        self.WMWidth = StringVar()  # 水印宽度, 提取水印用
        self.WMHeight = StringVar()  # 水印高度, 提取水印用

        self.RandomSeedWM.set('2333')
        self.RandomSeedDCT.set('6666')

    def OpenOriImg(self):  # 打开原始图片
        self.OriImgPath.set(filedialog.askopenfilename(
            title='Open original image', filetypes=[('PNG', '.png'), ('JPEG', '.jpg .jpeg')]))

    def OpenWMImg(self):  # 打开水印图片
        self.WMPath.set(filedialog.askopenfilename(
            title='Open watermark image', filetypes=[('PNG', '.png'), ('JPEG', '.jpg .jpeg')]))

    def OpenOutImg(self):  # 打开嵌入水印的图片
        self.OutImgPath.set(filedialog.askopenfilename(
            title='Open embedded image', filetypes=[('PNG', '.png'), ('JPEG', '.jpg .jpeg')]))

    def EmbedImg(self):
        if self.OriImgPath.get() == '':
            showerror('Error', 'Please choose an original image')
            return
        if self.WMPath.get() == '':
            showerror('Error', 'Please choose an watermark image')
            return
        if (not self.RandomSeedDCT.get().isdigit()) or (not self.RandomSeedWM.get().isdigit()):
            showerror('Error', 'Please input Seed1 and Seed2 correctly')
            return

        OutputPath = './Embedded_Image.png'

        try:
            bwm = WaterMark(int(self.RandomSeedWM.get()),
                            int(self.RandomSeedDCT.get()))
            bwm.ReadOriImg(self.OriImgPath.get())
            bwm.ReadWM(self.WMPath.get())
            bwm.Embed(OutputPath)
        except Exception:
            showerror('Error', 'Unknown Error')
            return

        showinfo('OK', 'Embedded image saved as Embedded_Image.png')
        return

    def ExtractImg(self):
        if self.OutImgPath.get() == '':
            showerror('Error', 'Please choose an embedded image')
            return
        if (not self.RandomSeedDCT.get().isdigit()) or (not self.RandomSeedWM.get().isdigit()):
            showerror('Error', 'Please input seed1 And seed2 correctly')
            return
        if (not self.WMHeight.get().isdigit()) or (not self.WMWidth.get().isdigit()):
            showerror('Error', 'Please input watermark size correctly')
            return

        OutputPath = './Extracted_WaterMark.png'

        try:
            bwm = WaterMark(int(self.RandomSeedWM.get()), int(
                self.RandomSeedDCT.get()), (int(self.WMWidth.get()), int(self.WMHeight.get())))
            bwm.Extract(self.OutImgPath.get(), OutputPath)
        except Exception:
            showerror('Error', 'Unknown Error')
            return

        showinfo('OK', 'Extracted watermark saved as Extracted_WaterMark.png')
        return

    def InitWindow(self):  # 初始化窗口
        # 原始图片
        Label(self.window, text='Original image:').grid(row=0, column=0)
        Entry(self.window, textvariable=self.OriImgPath).grid(row=0, column=1)
        Button(self.window, text='Open original image',
               command=self.OpenOriImg).grid(row=0, column=2)

        # 水印图片
        Label(self.window, text='Watermark image:').grid(row=1, column=0)
        Entry(self.window, textvariable=self.WMPath).grid(row=1, column=1)
        Button(self.window, text='Open Watermark image',
               command=self.OpenWMImg).grid(row=1, column=2)

        # 嵌入水印的图片
        Label(self.window, text='Embedded image:').grid(row=2, column=0)
        Entry(self.window, textvariable=self.OutImgPath).grid(row=2, column=1)
        Button(self.window, text='Open embedded image',
               command=self.OpenOutImg).grid(row=2, column=2)

        # 种子, 水印的大小
        Label(self.window, text='Seed1:').grid(row=3, column=0)
        Entry(self.window, textvariable=self.RandomSeedWM).grid(row=3, column=1)
        Label(self.window, text='Seed2:').grid(row=4, column=0)
        Entry(self.window, textvariable=self.RandomSeedDCT).grid(row=4, column=1)
        Label(self.window, text='WaterMark width:').grid(row=5, column=0)
        Entry(self.window, textvariable=self.WMWidth).grid(row=5, column=1)
        Label(self.window, text='WaterMark height:').grid(row=6, column=0)
        Entry(self.window, textvariable=self.WMHeight).grid(row=6, column=1)

        # 嵌入水印, 提取水印
        Button(self.window, text='Embed',
               command=self.EmbedImg).grid(row=7, column=0)
        Button(self.window, text='Extract',
               command=self.ExtractImg).grid(row=7, column=1)

    def mainloop(self):  # 启动窗口
        self.window.mainloop()


if __name__ == '__main__':
    RootWindow = GUI('BlinkWaterMark', '500x250')
    RootWindow.InitWindow()
    RootWindow.mainloop()
