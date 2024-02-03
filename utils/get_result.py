
import torch
import torch.nn.functional as F
import torch.utils.data as data

import os
import csv
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.general import all_label_color_dict,feature_color_list
from utils.evaluate_tool import Evaluate_Matrix
from dataloaders.dataset import DatasetFromFolder

def get_test(opt,if_eval=True):
    device = torch.device(opt.device)
    if opt.path_test_model is None:
        path_test_model = opt.save_model_root / f'{opt.extra_word}_s_{opt.epoch_test}.pth'
    else:
        path_test_model=opt.path_test_model
    path_save_yaml = opt.save_model_root / f'{opt.extra_word}_{opt.epoch_test}.yaml'
    path_save_fig = opt.save_result_root/ opt.file_name / 'figs' / f'{opt.extra_word}_{opt.epoch_test}.jpg'
    path_save_test_img=opt.save_result_root/ opt.file_name / opt.extra_word
    if os.path.exists(path_save_test_img):
        shutil.rmtree(path_save_test_img)
    os.mkdir(path_save_test_img)

    model = torch.load(path_test_model)
    model.to(device)
    if if_eval:
        model.eval()

    test_set= DatasetFromFolder(opt, subset='test',get_name=True,if_transform=False)
    test_loader=data.DataLoader(test_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    label_color_list=all_label_color_dict[opt.dataset_i]

    select_feature_list=[]
    select_label_list=[]

    test_mat = Evaluate_Matrix(opt)
    with torch.no_grad():
        bar=tqdm(test_loader)
        for batch_data in bar:
            rs_img = batch_data[0].to(device)  # n*c*h*w
            lab_img = batch_data[1].to(device)  # n*h*w
            name_list = batch_data[2]

            out_img,out_feature = model(rs_img)  # n*c*h*w，n*C*h'*w'

            feature_c = out_feature.shape[1]
            img_n,_, img_h, img_w=rs_img.shape
            out_img=F.interpolate(out_img,size=(img_h,img_w),mode='bilinear')  # n*c*h*w
            out_feature=F.interpolate(out_feature,size=(img_h,img_w),mode='bilinear')  # n*C*h*w

            predict_img = torch.argmax(out_img, dim=1)  # n*h*w
            test_mat.update(predict_img, lab_img)

            lab_img = lab_img.cpu()
            predict_img = predict_img.cpu()
            out_feature=out_feature.cpu()

            select_num_in_batch=int(opt.feature_per_class*img_n/len(test_set))
            img_batch_np=np.zeros((img_n, img_h, img_w, 3))

            for class_i in range(opt.class_num):
                place_n,place_h,place_w=torch.where(lab_img==class_i)
                place_choice_n = place_n[:select_num_in_batch]
                place_choice_h = place_h[:select_num_in_batch]
                place_choice_w = place_w[:select_num_in_batch]

                select_feature= out_feature[place_choice_n, :, place_choice_h, place_choice_w].split(1)
                select_feature_list+=select_feature

                select_label= lab_img.cpu()[place_choice_n, place_choice_h, place_choice_w].split(1)
                select_label_list+=select_label
                img_batch_np[predict_img == class_i]=label_color_list[class_i]
            if opt.dataset_i==1:
                img_batch_np[lab_img.numpy()<0]=np.array([0,0,0])
            img_batch_np=img_batch_np.astype(np.uint8)  # n*h*w*c

            for img_i in range(img_n):
                img_pil=Image.fromarray(img_batch_np[img_i]).convert('RGB')
                img_save_name=name_list[img_i].split('.')[0]+'.jpg'
                img_pil.save(opt.save_result_root/ opt.file_name / opt.extra_word / img_save_name)

        select_feature_all=torch.cat(select_feature_list).reshape((-1, feature_c))
        select_label_all=torch.cat(select_label_list).reshape(-1)

    test_metrics = test_mat.get_metrics()
    test_miou = test_metrics['miou']
    test_fwiou = test_metrics['fwiou']
    test_f1 = test_metrics['f1']
    test_pa = test_metrics['pa']
    test_mpa = test_metrics['mpa']
    test_kappa = test_metrics['kappa']
    opt.logger.info(f'{opt.file_name} {opt.extra_word}_{opt.epoch_test} {opt.short_name}_{int(opt.label_rate*100)}  results:')
    opt.logger.info(f"mIoU[{test_miou:.4f}] fwiou[{test_fwiou:.4f}] F1[{test_f1:.4f}] "
                    f"pa[{test_pa:.4f}] mpa[{test_mpa:.4f}] kappa[{test_kappa:.4f}]")

    with open(path_save_yaml) as f:
        info_dict=yaml.safe_load(f)
        info_dict['-test_metrics']=test_metrics
        info_dict['-test_matrix']=test_mat.confusion_matrix.tolist()
    with open(path_save_yaml,'w') as f:
        yaml.safe_dump(info_dict,f,sort_keys=False )

    tsne = TSNE(n_components=2, learning_rate=100)
    tsne_data = tsne.fit_transform(select_feature_all)

    plt.figure()
    plt.axis('off')
    for class_i in range(opt.class_num):
        select_place_i=select_label_all==class_i
        plt.scatter(tsne_data[select_place_i, 0], tsne_data[select_place_i, 1], c=
        feature_color_list[class_i], marker='.')

    plt.savefig(path_save_fig, dpi=400)

def get_process(opt):
    path_save_csv=opt.save_log_root / f'{opt.extra_word}_epoch.csv'
    epoch_list=[]
    process_list=[[],[],[]]
    with open(path_save_csv) as f:
        epoch_csv=csv.reader(f)
        header_list=next(epoch_csv)
        for row in epoch_csv:
            epoch_list.append(int(row[0]))
            process_list[0].append(float(row[1]))
            process_list[1].append(float(row[2]))
            process_list[2].append(float(row[3]))
    for header_i in range(len(header_list)-1):
        fig=plt.figure()
        plt.plot(epoch_list,process_list[header_i])
        fig.savefig(opt.save_result_root/ opt.file_name / 'figs' / f'{opt.extra_word}_{header_list[header_i+1]}.jpg')

