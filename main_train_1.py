
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler

import csv
import yaml
import copy
import datetime
import warnings
from tqdm import tqdm
from time import time
from pathlib import Path

from config import parse_opt
from utils.general import colorstr
from utils.defined_loss import cal_c_loss
from utils.evaluate_tool import Evaluate_Matrix
from utils.get_result import get_test,get_process
from dataloaders.dataset import DatasetFromFolder
from networks.segmind.Net import net

cudnn.benchmark = True
warnings.filterwarnings("ignore")

class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

def get_mask_tensor(h=512,w=512,mask_gap=16,mask_rate=0.75):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask_tensor_small=torch.randperm(h_gap_num*w_gap_num).float().reshape((h_gap_num,w_gap_num))
    divide_threshold=h_gap_num * w_gap_num * mask_rate
    mask_tensor_small[mask_tensor_small<divide_threshold]=0.0
    mask_tensor_small[mask_tensor_small>=divide_threshold]=1.0
    mask_tensor=F.interpolate(mask_tensor_small.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0)

    return mask_tensor

def get_batch_mask_tensor(nchw=(1,3,512,512),mask_gap=16,mask_rate=0.75):
    mask_tensor=torch.zeros((nchw[0],nchw[-2],nchw[-1]))
    for img_i in range(nchw[0]):
        mask_tensor[img_i]=get_mask_tensor(h=nchw[-2],w=nchw[-1],mask_gap=mask_gap,mask_rate=mask_rate)
    return mask_tensor.unsqueeze(1)

def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()

def generate_u_data(img_u_w,img_u_s, lab_u, logits_u,entropy_u, opt):
    batch_size, _, im_h, im_w = img_u_w.shape
    device = torch.device(opt.device)

    new_img_w = []
    new_img_s = []
    new_lab = []
    new_logits = []
    new_entropy = []

    for i in range(batch_size):
        mix_mask = generate_class_mask(lab_u[i]).to(device)

        new_img_w.append((img_u_w[i] * mix_mask + img_u_w[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_img_s.append((img_u_s[i] * mix_mask + img_u_s[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_lab.append((lab_u[i] * mix_mask + lab_u[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits_u[i] * mix_mask + logits_u[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_entropy.append((entropy_u[i] * mix_mask + entropy_u[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_img_w, new_img_s, new_lab, new_logits,entropy_u = \
        torch.cat(new_img_w), torch.cat(new_img_s), torch.cat(new_lab), torch.cat(new_logits),torch.cat(new_entropy)
    return new_img_w, new_img_s, new_lab.long(), new_logits,entropy_u

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

def get_pseudo(t_model,rs_l_w,rs_u_w,hw,n_l):
    t_pred_all, _ = t_model.model(torch.cat((rs_l_w, rs_u_w), dim=0))  # 2n*4*h*w->2n*c*h'*w'
    h_w_ = t_pred_all.shape[-2:]  # h'*w'
    t_pred_all = F.interpolate(t_pred_all, size=hw, mode='bilinear', align_corners=True)  # 2n*c*h*w
    t_prob_all = torch.softmax(t_pred_all, dim=1)  # 2n*c*h*w
    pseudo_logit, pseudo_label = torch.max(t_prob_all[n_l:], dim=1)  # n*h*w,n*h*w
    t_entropy_all = torch.sum(-t_prob_all * torch.log(t_prob_all + 1e-8), dim=1)  # 2n*h*w
    return h_w_,pseudo_logit,pseudo_label,t_entropy_all

def get_data(device,train_loader_l_iter,train_loader_l,train_loader_u_iter,train_loader_u,t_model):
    try:
        rs_l_w, rs_l_s, lab_l = next(train_loader_l_iter)  # n*3*h*w,n*3*h*w,n*h*w
    except:
        train_loader_l_iter = iter(train_loader_l)
        rs_l_w, rs_l_s, lab_l = next(train_loader_l_iter)
    rs_l_w, rs_l_s, lab_l = rs_l_w.to(device), rs_l_s.to(device), lab_l.to(device)

    try:
        rs_u_w, rs_u_s, _ = next(train_loader_u_iter)  # n*3*h*w,n*3*h*w
    except:
        train_loader_u_iter = iter(train_loader_u)
        rs_u_w, rs_u_s, _ = next(train_loader_u_iter)
    rs_u_w, rs_u_s = rs_u_w.to(device), rs_u_s.to(device)

    n_l = lab_l.size()[0]  # n
    hw = lab_l.size()[-2:]  # h*w

    with torch.no_grad():
        h_w_, pseudo_logit, pseudo_label, t_entropy_all = get_pseudo(t_model, rs_l_w, rs_u_w, hw, n_l)

        rs_u_w, rs_u_s, pseudo_label, pseudo_logit, t_entropy_all[n_l:] = \
            generate_u_data(rs_u_w, rs_u_s, pseudo_label, pseudo_logit, t_entropy_all[n_l:], opt)

        lab_all = torch.cat((lab_l, pseudo_label), dim=0)
        lab_u_reli = pseudo_logit.ge(opt.pseudo_threshold).float()
        mask_all = torch.cat((lab_l >= 0, lab_u_reli), dim=0)
        rs_all_s = torch.cat((rs_l_s, rs_u_s), dim=0)
        rs_all_w = torch.cat((rs_l_w, rs_u_w), dim=0)
    return train_loader_l_iter,train_loader_u_iter, lab_all, mask_all, rs_all_s,rs_all_w,hw,n_l,h_w_,t_entropy_all

def get_loss_lab(s_pred_all,lab_all,cal_loss_l):

    loss_l = cal_loss_l(s_pred_all, lab_all)
    return loss_l

def get_loss_e(s_prob_all,cal_loss_e,t_entropy_all):
    s_entropy_all = torch.sum(-s_prob_all * torch.log(s_prob_all + 1e-8), dim=1)  # 学生预测图像的熵，2n*h*w
    loss_e = cal_loss_e(s_entropy_all, t_entropy_all)
    return loss_e

def get_loss_r_rsc(rs_all_w,device,s_model,hw,cal_loss_r,cal_loss_rsc,lab_all,r_model=None):
    """rs_all_w:2n*3*h*w"""
    loss_r,loss_rsc=0.0,0.0
    nchw=rs_all_w.size()
    with torch.no_grad():
        mask_tensor = get_batch_mask_tensor(nchw=nchw, mask_rate=opt.mask_rate_end).to(device)  # 2n*1*h*w


    r_pred_all, _, r_img_all = s_model(rs_all_w * mask_tensor, mode='r',model_r=r_model,mask=mask_tensor)  # 2n*c*h*w

    r_img_all = F.interpolate(r_img_all, size=hw, mode='bilinear', align_corners=True)  # 2n*3*h*w

    if opt.lambda_r != 0:
        loss_r = cal_loss_r(r_img_all.permute(0, 2, 3, 1)[~mask_tensor.bool().squeeze(1)],
            rs_all_w[:,:3,:,:].permute(0, 2, 3, 1)[~mask_tensor.bool().squeeze(1)])

    if opt.lambda_rsc != 0:
        loss_rsc = cal_loss_rsc(r_pred_all.permute(0, 2, 3, 1)[~mask_tensor.bool().squeeze(1)],
                                lab_all[~mask_tensor.bool().squeeze(1)])
    return loss_r,loss_rsc

def get_loss_c(lab_all,h_w_,s_prob_all,cal_loss_c,s_feat_all,opt,memory_bank_list,queue_size,queue_ptr_list):
    with torch.no_grad():
        lab_all_small = F.interpolate(lab_all.float().unsqueeze(1), size=h_w_, mode='nearest')  # 2n*h*w->2n*1*h'*w'
        s_prob_all = F.interpolate(s_prob_all, size=h_w_, mode='bilinear', align_corners=True)  # 2n*c*h'*w'

    loss_c = cal_loss_c(feat=s_feat_all, lab=lab_all_small.long().squeeze(1), prob=s_prob_all,opt=opt,
                        memory_bank_list=memory_bank_list, queue_size=queue_size, queue_ptr_list=queue_ptr_list)
    return loss_c

def get_val(s_model,validate_loader,device,validate_mat,in_channel=3):
    with torch.no_grad():
        s_model.eval()
        for batch_data in validate_loader:
            rs_img = batch_data[0].to(device)
            lab_img = batch_data[1].to(device)
            out_img, _ = s_model(rs_img)  # n*c*h'*w'
            # n*c*h'*w'->n*c*h*w
            out_img = F.interpolate(out_img, size=lab_img.size()[-2:], mode='bilinear', align_corners=True)
            predict_img = torch.argmax(out_img, dim=1)  # n*h*w
            validate_mat.update(predict_img, lab_img)
    validate_metrics = validate_mat.get_metrics()
    validate_miou = validate_metrics['miou']
    validate_mat.reset()
    return validate_metrics,validate_miou,validate_mat

def train(opt):
    device = torch.device(opt.device)

    path_save_csv = opt.save_log_root / f'{opt.extra_word}_epoch.csv'

    train_set_l = DatasetFromFolder(opt, subset='train', label_rate=opt.label_rate,labeled=True,if_transform=True,get_mask=False)
    train_set_u = DatasetFromFolder(opt, subset='train', label_rate=opt.label_rate,labeled=False,if_transform=True,get_mask=False)
    validate_set = DatasetFromFolder(opt, subset='validate',label_rate=1,labeled=True, if_transform=False,get_mask=False)
    train_loader_l=data.DataLoader(train_set_l, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True,drop_last=True)
    train_loader_u=data.DataLoader(train_set_u, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True,drop_last=True)
    validate_loader=data.DataLoader(validate_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False,drop_last=False)
    iters_per_epoch=len(train_loader_l)

    if opt.epoch_start==0:
        s_model=net(num_classes=opt.class_num,in_channel=3)
        t_model=EMA(s_model, 0.99)

        best_epoch=0
        best_fitness = 0.0
        learn_rate = opt.learn_rate
        epoch_csv=csv.writer(open(path_save_csv, 'w', newline=''))
        epoch_csv.writerow(['epoch', 'learn_rate', 'loss_validate_epoch', 'miou'])
    else:
        s_model=torch.load(opt.save_model_root / f'{opt.extra_word}_s_{opt.epoch_start}.pth')
        t_model = EMA(None, 0.99)
        t_model.model=torch.load(opt.save_model_root / f'{opt.extra_word}_t_{opt.epoch_start}.pth')

        info_dict = yaml.safe_load(open(opt.save_model_root / f'{opt.extra_word}_{opt.epoch_start}.yaml'))
        best_epoch=int(info_dict['-best_epoch'])
        best_fitness=float(info_dict['-best_fitness'])
        learn_rate=float(info_dict['-learn_rate'])
        epoch_csv = csv.writer(open(path_save_csv, 'a', newline=''))
    s_model.to(device)
    t_model.model.to(device)

    cal_loss_l=nn.CrossEntropyLoss(ignore_index=-1)
    cal_loss_e = nn.MSELoss()
    cal_loss_r=nn.MSELoss()
    cal_loss_rsc=nn.CrossEntropyLoss(ignore_index=-1)
    cal_loss_c=cal_c_loss

    optimizer_s=optim.AdamW(s_model.parameters(), lr=learn_rate, betas=(0.9, 0.99),weight_decay=opt.weight_delay)
    lr_scheduler=PolyLR(optimizer_s, opt.epoch_end*iters_per_epoch, last_epoch=-1)
    validate_mat = Evaluate_Matrix(opt)

    memory_bank_list = []
    queue_size = []
    queue_ptr_list = []
    for i in range(opt.class_num):
        memory_bank_list.append([torch.zeros(0, 256)])
        queue_size.append(opt.bank_size)
        queue_ptr_list.append(torch.zeros(1, dtype=torch.long))

    opt.start_time=time()/60
    opt.used_time=0
    opt.logger.info(f"{colorstr('blue', 'start')} training")

    for epoch in range(opt.epoch_start + 1, opt.epoch_end + 1):
        s_model.train()
        t_model.model.train()
        loss_epoch = 0
        train_loader_l_iter=iter(train_loader_l)
        train_loader_u_iter=iter(train_loader_u)
        bar=tqdm(range(iters_per_epoch))
        for _ in bar:
            loss_l,loss_e,loss_c,loss_r,loss_rsc=0.0,0.0,0.0,0.0,0.0
            train_loader_l_iter, train_loader_u_iter, lab_all, mask_all, rs_all_s, rs_all_w,hw,n_l,h_w_,t_entropy_all = \
                get_data(device,train_loader_l_iter,train_loader_l,train_loader_u_iter,train_loader_u,t_model)

            s_pred_all,s_feat_all=s_model(rs_all_s)  # 2n*4/3*h*w -> 2n*c*h'*w',2n*C*h'*w'
            s_pred_all=F.interpolate(s_pred_all, size=hw, mode='bilinear', align_corners=True)  # 2n*c*h*w
            s_prob_all = torch.softmax(s_pred_all, dim=1)  # 2n*c*h*w

            if opt.lambda_l != 0:
                loss_l=get_loss_lab(s_pred_all,lab_all,cal_loss_l)

            if opt.lambda_e != 0:
                loss_e=get_loss_e(s_prob_all,cal_loss_e,t_entropy_all)

            if (opt.lambda_r != 0 or opt.lambda_rsc!=0) and epoch <= opt.epoch_pre:
                loss_r,loss_rsc=get_loss_r_rsc(rs_all_w,device,s_model,hw,cal_loss_r,cal_loss_rsc,lab_all)

            if opt.lambda_c!=0:
                loss_c=get_loss_c(lab_all,h_w_,s_prob_all,cal_loss_c,s_feat_all,opt,memory_bank_list,queue_size,queue_ptr_list)

            optimizer_s.zero_grad()
            loss=opt.lambda_l*loss_l+opt.lambda_e*loss_e+opt.lambda_rsc*loss_rsc+opt.lambda_r*loss_r+opt.lambda_c*loss_c
            loss.backward()
            optimizer_s.step()

            with torch.no_grad():
                t_model.update(s_model)

            loss_batch = loss.item()
            loss_epoch+=loss_batch
            lr_scheduler.step()
            batch_words=f'epoch[{epoch:3d}/{opt.epoch_end:3d}]\tloss[{loss_batch:.2e}]\t' \
                        f'memory[{torch.cuda.memory_reserved() / 1E9:2.1f}G]'
            bar.set_description(batch_words)

        validate_metrics,validate_miou,validate_mat=get_val(s_model,validate_loader,device,validate_mat,in_channel=3)

        used_time=time()/60-opt.start_time
        epoch_time=used_time-opt.used_time
        opt.used_time=used_time
        learn_rate=optimizer_s.param_groups[0]['lr']
        epoch_csv.writerow([epoch, learn_rate, loss_epoch, validate_miou])
        except_used_time=used_time+epoch_time*(opt.epoch_end - epoch)
        end_time = (datetime.datetime.now() + datetime.timedelta(minutes=epoch_time * (opt.epoch_end - epoch))).strftime('%H:%M')

        info_dict={
            '-epoch':epoch,
            '-used_time': used_time,
            '-best_epoch':best_epoch,
            '-best_fitness':best_fitness,
            '-learn_rate':learn_rate,
            '-validate_metrics':validate_metrics,
            '-test_metrics':None,
            '-test_matrix':None
        }

        if validate_miou>best_fitness:
            best_epoch=epoch
            best_fitness=validate_miou
            torch.save(s_model, opt.save_model_root / f'{opt.extra_word}_s_best.pth')
            torch.save(t_model.model, opt.save_model_root / f'{opt.extra_word}_t_best.pth')
            with open(opt.save_model_root / f'{opt.extra_word}_best.yaml', 'w') as f:
                yaml.safe_dump(info_dict, f,sort_keys=False )
        if epoch % opt.epoch_gap == 0:
            torch.save(s_model, opt.save_model_root / f'{opt.extra_word}_s_{epoch}.pth')
            torch.save(t_model.model, opt.save_model_root / f'{opt.extra_word}_t_{epoch}.pth')
            with open(opt.save_model_root / f'{opt.extra_word}_{epoch}.yaml', 'w') as f:
                yaml.safe_dump(info_dict, f)

        epoch_words=f"\ttime[{used_time//60:.0f}:{used_time%60:.0f}{colorstr('bold','|')}{except_used_time//60:.0f}:" \
                    f"{except_used_time%60:.0f}{colorstr('bold','|')}{end_time:s}]\t" \
                    f"learn_rate[{learn_rate:.2e}]\tloss_epoch[{loss_epoch:.2e}]\t" \
                    f"mIoU[{colorstr('bold', f'{validate_miou:.4f}')}]\tbest[{best_epoch} {best_fitness:.4f}]"
        opt.logger.info(epoch_words)

if __name__ == '__main__':

    file_name=Path(__file__).resolve().stem

    opt = parse_opt()

    import_run_para=f'file_name:[{file_name}]\tdataset:[{opt.dataset_name}]\tlabel rate:[{opt.label_rate}]\t'\
        f'extra word:[{opt.extra_word}]\tepoch:[{opt.epoch_start}~{opt.epoch_end}]\tbatch size:[{opt.batch_size}]'
    opt.logger.info(import_run_para)

    opt.file_name=file_name
    opt.save_model_root=opt.root / 'checkpoints' / file_name
    opt.path_test_model = opt.save_model_root / f'{opt.extra_word}_s_{opt.epoch_test}.pth'
    opt.save_log_root=opt.root / 'logs' / file_name

    opt.save_model_root.mkdir(parents=True, exist_ok=True)
    opt.save_log_root.mkdir(parents=True, exist_ok=True)
    (opt.save_result_root/ opt.file_name / 'figs').mkdir(parents=True, exist_ok=True)
    (opt.save_result_root/ opt.file_name / opt.extra_word).mkdir(parents=True, exist_ok=True)

    train(opt)
    opt.logger.info(f"used time:{colorstr('bold',f'{opt.used_time//60:.0f}hours {opt.used_time%60:.1f} mins')}")
    opt.logger.info(f"{colorstr('blue','start')} testing")
    get_test(opt)
    get_process(opt)



