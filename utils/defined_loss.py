
import torch
import torch.nn.functional as F

import numpy as np

@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    keys_num = keys.shape[0]
    ptr = int(queue_ptr)
    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size #
    else:
        ptr = (ptr + keys_num) % queue_size
    queue_ptr[0] = ptr

def get_negative_feat(samp_num, memo_list,opt,feat_num):
    negative_feat_all = torch.zeros((opt.num_query,opt.num_negative,feat_num))
    for i in range(samp_num.shape[0]):
        negative_feat_i_list=[]
        for j in range(samp_num.shape[1]):
            if memo_list[j][0].shape[0]==0:
                continue
            negative_index = np.random.randint(low=0,high=memo_list[j][0].shape[0],size=int(samp_num[i, j])).tolist()
            negative_feat_i_list.append(memo_list[j][0][negative_index])
        negative_feat_i=torch.cat(negative_feat_i_list)
        negative_num =negative_feat_i.shape[0]
        negative_feat_all[i,:negative_num]=negative_feat_i
    return negative_feat_all

def cal_c_loss(feat,lab,prob,opt,memory_bank_list,queue_size,queue_ptr_list):
    device = torch.device(opt.device)
    loss_c=torch.tensor(0.0).to(device)
    feat_num = feat.shape[1]
    feat=feat.permute(0,2,3,1)  # n*C*h*w->n*h*w*C

    valid_class_batch_list=[]
    feat_mean_batch_list = []
    feat_hard_batch_list = []
    feat_mean_set_tensor = torch.zeros((opt.class_num, feat_num))  # c*C

    for class_i in range(opt.class_num):  # 0~c-1
        lab_i_place = lab==class_i  # n*h*w
        if torch.sum(lab_i_place)==0:
            continue
        valid_class_batch_list.append(class_i)

        prob_i = prob[:, class_i, :, :]  # n*h*w
        feat_i_hard_mask = (prob_i < opt.query_threshold) * lab_i_place  # n*h*w

        dequeue_and_enqueue(keys=feat[lab_i_place],queue=memory_bank_list[class_i],queue_ptr=queue_ptr_list[class_i],queue_size=queue_size[class_i])
        feat_mean_batch_list.append(torch.mean(feat[lab_i_place], dim=0, keepdim=True))  # 1*C
        feat_hard_batch_list.append(feat[feat_i_hard_mask])  # hnum*C
        if len(memory_bank_list[class_i][0]) > 0:
            feat_mean_set_tensor[class_i]=torch.mean(memory_bank_list[class_i][0], dim=0)

    valid_class_num = len(valid_class_batch_list)  # cnum

    for v_class_i in range(valid_class_num):  # 0~cnum-1
        v_class_kind=valid_class_batch_list[v_class_i]
        if len(feat_hard_batch_list[v_class_i]) > 0:
            feat_hard_idx = torch.randint(len(feat_hard_batch_list[v_class_i]), size=(opt.num_query,))
            query_feat = feat_hard_batch_list[v_class_i][feat_hard_idx]
        else:
            continue

        with torch.no_grad():
            feat_mean_sim = torch.cosine_similarity(feat_mean_batch_list[v_class_i], feat_mean_set_tensor.to(device), dim=1)
            feat_mean_sim=torch.cat((feat_mean_sim[:v_class_kind],feat_mean_sim[v_class_kind+1:]))
            negative_sample_prob=torch.softmax(feat_mean_sim,dim=0)
            negative_sample_prob=torch.tensor(negative_sample_prob[:v_class_kind].tolist()+[0]+negative_sample_prob[v_class_kind+1:].tolist())
            negative_num_sampler = torch.distributions.categorical.Categorical(probs=negative_sample_prob)
            sample_class = negative_num_sampler.sample(sample_shape=[opt.num_query, opt.num_negative])
            sample_class_num=torch.stack([(sample_class == c).sum(1) for c in range(opt.class_num)], dim=1)

            negative_feat = get_negative_feat(sample_class_num, memory_bank_list,opt,feat_num).to(device)
            positive_feat = feat_mean_batch_list[v_class_i].unsqueeze(0).repeat(opt.num_query, 1, 1)
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = torch.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2).to(device)
        loss_c += F.cross_entropy(seg_logits / opt.temperature, torch.zeros(opt.num_query).long().to(device))

    return loss_c / valid_class_num


