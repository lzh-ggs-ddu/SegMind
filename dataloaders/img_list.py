
import os
import random

def txt2list(txt_path):
    """读取并返回遥感图像和标签图像的绝对路径列表，txt_path是txt文件的绝对路径"""
    rs_list=[]  # 初始化遥感图像名字列表
    lab_list=[]  # 初始化标签图像名字列表
    txt_file=open(txt_path,'r')
    for line in txt_file:  # 遍历每一行
        rs_list.append(line.split(',')[0])
        lab_list.append(line.split(',')[1].replace('\n',''))
    return rs_list,lab_list

def read_image_list(opt,subset='train'):
    """读取遥感图像名字列表和标签图像名字列表，subset为子数据集名字(train,validate,test)"""
    return txt2list(f'{opt.dataset_txt_root}/{opt.short_name}_{subset}.txt')  # 返回遥感图像绝对路径列表和标签图像绝对路径列表

def list2txt(rs_path,rs_list,lab_path,lab_list,txt_path,txt_mode='w'):
    """将遥感图像绝对路径列表rs_path+rs_list和标签图像绝对路径列表lab_path+lab_list写入绝对路径txt_path
    每行是遥感图像和标签图像绝对路径，要求遥感图像名字列表和标签图像名字列表要一一对应"""
    random.seed(0)
    random_index=list(range(len(rs_list)))
    random.shuffle(random_index)
    rs_list=[rs_list[r_i] for r_i in random_index]
    lab_list=[lab_list[r_i] for r_i in random_index]

    txt_file = open(txt_path, txt_mode)  # 去掉了encoding='utf-8'，要是出差了再改
    for i in range(len(rs_list)):  # 遍历每一个图像
        txt_file.write(f'{rs_path}/{rs_list[i]},{lab_path}/{lab_list[i]}\n')
    txt_file.flush()
    txt_file.close()

def make_image_list(opt,subset='train'):
    """生成数据集绝对路径列表txt，subset为子数据集名字(train,validate,test)"""
    dataset_path=os.path.join(opt.dataset_root,opt.dataset_name).replace('\\','/')  # 数据集绝对路径
    if opt.dataset_name=='LoveDA':
        subset_from=opt.subset_from_dict[subset]  # 数据集来源名字
        rs_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集遥感图像路径
        lab_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集标签图像路径
        rs_id_list = [rs_i for rs_i in os.listdir(rs_path) if rs_i.endswith("sat.png")]  # 子集遥感图像名字列表
        lab_id_list = [rs_i for rs_i in os.listdir(lab_path)  if rs_i.endswith('lab.png')]  # 子集遥感图像名字列表
        list2txt(rs_path, rs_id_list, lab_path, lab_id_list, f'{opt.dataset_txt_root}/{opt.short_name}_{subset}.txt')

    elif opt.dataset_name=='DeepGlobe_LandCover':
        subset_from=opt.subset_from_dict[subset]  # 数据集来源名字
        rs_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集遥感图像路径
        lab_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集标签图像路径
        rs_id_list = [rs_i for rs_i in os.listdir(rs_path) if rs_i.endswith("sat.png")]  # 子集遥感图像名字列表
        lab_id_list = [rs_i for rs_i in os.listdir(lab_path)  if rs_i.endswith('lab.png')]  # 子集遥感图像名字列表
        list2txt(rs_path, rs_id_list, lab_path, lab_id_list, f'{opt.dataset_txt_root}/{opt.short_name}_{subset}.txt')

    elif opt.dataset_name=='postdam':
        subset_from=opt.subset_from_dict[subset]  # 数据集来源名字
        rs_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集遥感图像路径
        lab_path = os.path.join(dataset_path, subset_from).replace('\\', '/')  # 子集标签图像路径
        rs_id_list = [rs_i for rs_i in os.listdir(rs_path) if rs_i.endswith("sat.png")]  # 子集遥感图像名字列表
        lab_id_list = [rs_i for rs_i in os.listdir(lab_path)  if rs_i.endswith('lab.png')]  # 子集遥感图像名字列表
        list2txt(rs_path, rs_id_list, lab_path, lab_id_list, f'{opt.dataset_txt_root}/{opt.short_name}_{subset}.txt')



