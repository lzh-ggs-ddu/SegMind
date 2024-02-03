"""
File Name  :dataset_3
Author     :Li ZhengHong
Create Date:2022/11/8
----------------------------------------
Change Date:2022/11/8
Description:在dataset_2的基础上去除shift，强数据增强依据psmt
"""
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from dataloaders.img_list import read_image_list
from utils.general import all_label_color_dict

def rotate(image, label,angle=10):
    """随机旋转-angle到angle度"""
    h, w, _ = image.shape
    rot_angle = random.randint(-angle, angle)
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)  # , borderMode=cv2.BORDER_REFLECT)
    label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)
    return image, label

def crop(image, label, crop_size):
    """随机裁剪图像，裁剪后图像形状为(crop_size[0],crop_size[1])"""
    if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
        crop_h, crop_w = crop_size
    elif isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        raise ValueError

    h, w, _ = image.shape
    if crop_h>h or crop_w>w:
        raise ValueError

    # Cropping
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    image = image[start_h:end_h, start_w:end_w]
    label = label[start_h:end_h, start_w:end_w]
    return image, label

def blur(image, label, sigma_max=1.5):
    """随机添加高斯噪声，水平和竖直方向标准差为sigma"""
    sigma = random.random() * sigma_max
    ksize = int(3.3 * sigma)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    return image, label

def flip(image, label,flip_p=0.5):
    """以flip_p的概率随机水平、竖直方向翻转"""
    if random.random() < flip_p:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()
    if random.random() < flip_p:
        image = np.flipud(image).copy()
        label = np.flipud(label).copy()
    return image, label

def resize(image, label,new_size):
    """对图像缩放,base_size为缩放后尺寸，"""
    if isinstance(new_size, int):
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((new_size, new_size), Image.BICUBIC))
        label = cv2.resize(label, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
        return image, label
    else:
        raise ValueError

def get_shift_crop_place(h=1024,w=1024,crop_size=512,min_iou=0.1,max_iou=0.9,max_try=10):
    """获取随机偏移裁剪位置，h、w是图像高和宽，crop_size是裁剪窗口宽度，min_iou是偏移裁剪最小交并比,max_iou是最大交并比,max_try是最大尝试数量"""
    for try_i in range(max_try):
        place_1_h=random.randint(0,h-crop_size)  # 裁剪窗口1的h位置
        place_1_w=random.randint(0,w-crop_size)  # 裁剪窗口1的w位置
        place_2_h=random.randint(0,h-crop_size)  # 裁剪窗口2的h位置
        place_2_w=random.randint(0,w-crop_size)  # 裁剪窗口2的w位置
        intersection_h=min(place_1_h+crop_size-place_2_h,place_2_h+crop_size-place_1_h)
        intersection_w=min(place_1_w+crop_size-place_2_w,place_2_w+crop_size-place_1_w)
        iou=(intersection_h*intersection_w)/(2 * crop_size**2-intersection_h*intersection_w)
        if min_iou <= iou <= max_iou:
            return [place_1_h,place_1_w,place_2_h,place_2_w]
    iou=random.uniform(min_iou,max_iou)  # 随机生成交并比
    shift_length=int((1-iou)*crop_size)  # 偏移像素数
    place_1_h = random.randint(0, h - crop_size-shift_length)  # 裁剪窗口1的h位置
    place_1_w = random.randint(0, w - crop_size-shift_length)  # 裁剪窗口1的w位置
    if random.random()>0.5:  # 一半概率横向覆盖，一半概率纵向覆盖
        place_2_h=place_1_h+shift_length
        place_2_w=place_1_w
    else:
        place_2_h=place_1_h
        place_2_w=place_1_w+shift_length
    return [place_1_h,place_1_w,place_2_h,place_2_w]

def get_relative_place(place_1_h,place_1_w,place_2_h,place_2_w,crop_size):
    """依据裁剪位置和裁剪窗口尺寸计算相交区域在两个裁剪图像的的相对位置"""
    relative_place_1_h=max(place_2_h-place_1_h,0)
    relative_place_1_w=max(place_2_w-place_1_w,0)
    relative_place_2_h=max(place_1_h-place_2_h,0)
    relative_place_2_w=max(place_1_w-place_2_w,0)
    intersection_h = min(place_1_h + crop_size - place_2_h, place_2_h + crop_size - place_1_h)
    intersection_w = min(place_1_w + crop_size - place_2_w, place_2_w + crop_size - place_1_w)
    return [relative_place_1_h,relative_place_1_w],[relative_place_2_h,relative_place_2_w],[intersection_h,intersection_w]

class DatasetFromFolder(data.Dataset):
    def __init__(self,opt,label_rate=1,subset='train',labeled=True,get_name=False,if_transform=False,get_mask=False):
        """数据集"""
        super(DatasetFromFolder, self).__init__()
        self.label_rate=label_rate  # 标签率
        self.subset=subset  # 子数据集名字(train,valid,test)
        self.labeled=labeled  # True选取有标签部分，False选取无标签部分
        self.get_name=get_name  # 是否返回图像名字
        self.if_transform=if_transform  # 是否进行数据增强训练数据集为True，验证、测试数据集为False
        self.get_mask=get_mask  # if_transform=True的情况下，是否返回mask

        self.class_num=opt.class_num  # 标签类别数
        # 遥感图像绝对路径列表和标签图像绝对路径列表
        self.rs_list,self.lab_list=read_image_list(opt,subset=self.subset)  # 遥感和标签图像名字列表
        self.label_color=all_label_color_dict[opt.dataset_i]  # 网络标签和颜色对应列表
        self.resize_length=opt.resize_length  # resize后的图像尺寸，即输入网络的图像尺寸
        #self.large_size=opt.resize_length*2  # 初始化large_size,if_transform=True的情况下，先resize到大尺寸再裁剪

    def __getitem__(self, index):
        """返回rs_list、lab_list第index个元素对应遥感图像、标签图像"""
        if not self.labeled:  # 如果是返回无标签部分则从后往前选取
            index=-index
        ####################定义数据转换####################
        inter_nearest = transforms.InterpolationMode.NEAREST  # 最近邻插值
        if self.if_transform:  # 如果进行数据增强（训练，非验证,非测试）
            #self.large_size = int(self.resize_length*random.uniform(1,1.5))  # 随机缩放图像
            self.large_size = self.resize_length  # 随机缩放图像
            rot_degree = random.choice([0, 90, 180, 270])  # 随机旋转角度

            rs_enhance_list=[
                transforms.Resize((self.large_size, self.large_size)),  # 缩放
                #transforms.RandomCrop((self.resize_length, self.resize_length)),  # 裁剪
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree)),  # 随机旋转
                # 随机水平、垂直平移-0.1~0.1，随机缩放到0.9~1.1
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，[0,1]->[-1,1]
            ]
            lab_enhance_list = [
                transforms.Resize((self.large_size, self.large_size), interpolation=inter_nearest),   # 缩放
                #transforms.RandomCrop((self.resize_length, self.resize_length)),  # 裁剪
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree),interpolation=inter_nearest),  # 随机旋转
                # 随机水平、垂直平移-0.1~0.1，随机缩放到0.9~1.1
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),fill=-1),
                transforms.ToTensor()  # 转换为张量，[0,255]->[0,1]
            ]
        else:  # （验证、测试）
            rs_enhance_list=[
                transforms.Resize((self.resize_length,self.resize_length)),  # 缩放
                # transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，[0,1]->[-1,1]
            ]
            lab_enhance_list=[
                transforms.Resize((self.resize_length, self.resize_length), interpolation=inter_nearest),  # 最近邻插值
                transforms.ToTensor()  # 转换为张量，[0,255]->[0,1]
            ]

        transform_rs = transforms.Compose(rs_enhance_list)  # 遥感图像数据增强
        transform_lab = transforms.Compose(lab_enhance_list)  # 标签图像数据增强

        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        transforms_s=transforms.Compose([  # 强数据增强
            # 随机改变亮度、对比度、饱和度、色相
            transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.1)],p=0.8),
            # 随机转灰度图
            transforms.RandomGrayscale(p=0.1),
            # 随机高斯滤波
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],p=0.1)
        ])
        ####################.####################

        ####################读取数据并转换####################
        rs_data=Image.open(self.rs_list[index]).convert('RGB')  # 遥感图像数据
        lab=Image.open(self.lab_list[index]).convert('RGB')  # 标签图像数据

        trans_seed = np.random.randint(2147483647)  # 数据增强的随机数种子

        torch.manual_seed(trans_seed)
        torch.cuda.manual_seed(trans_seed)
        rs_data = transform_rs(rs_data)  # 遥感图像数据增强

        torch.manual_seed(trans_seed)
        torch.cuda.manual_seed(trans_seed)
        lab = (transform_lab(lab).squeeze(0) * 255).long()  # 标签图像数据增强，由于乘以255，标签取值范围0~255

        lab_data = torch.zeros(lab.shape[-2:])-1  # 初始化标签(lab_data h*w)
        for class_i in range(self.class_num):
            class_i_index=(lab.permute(1,2,0)==torch.tensor(self.label_color[class_i])).sum(dim=2)==3
            lab_data[class_i_index]=class_i
        lab_data=lab_data.long()

        if self.if_transform:  # 如果要进行转换，得到强数据增强数据(训练T，验证、测试F)
            rs_data_s = transforms_s(rs_data)

            rs_data = transforms.ToTensor()(rs_data)  # 转换为张量，[0,255]->[0,1]
            rs_data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(rs_data)  # 归一化，[0,1]->[-1,1]
            rs_data_s=transforms.ToTensor()(rs_data_s)  # 转换为张量，[0,255]->[0,1]
            rs_data_s=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(rs_data_s)  # 归一化，[0,1]->[-1,1]

            return rs_data,rs_data_s,lab_data

        rs_data = transforms.ToTensor()(rs_data)  # 转换为张量，[0,255]->[0,1]
        rs_data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(rs_data)  # 归一化，[0,1]->[-1,1]
        if self.get_name:  # 如果要额外返回图像名字（测试T，验证F）
            rs_path=self.rs_list[index].split(',')[0]
            rs_name=rs_path.split('/')[-1]
            return rs_data,lab_data,rs_name  # 额外返回图像名字

        return rs_data,lab_data
        ####################.####################

    def __len__(self):
        """数据集元素数量"""
        if self.labeled:  # 有监督数据集
            return int(len(self.rs_list)*self.label_rate)
        else:  # 无监督数据集，除有监督外的图像（即，若标签率为20%，返回80%的无监督数据）
            return len(self.rs_list)-int(len(self.rs_list)*self.label_rate)




