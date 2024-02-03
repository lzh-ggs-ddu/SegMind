
import torch

class Evaluate_Matrix(object):

    def __init__(self, opt):
        self.opt=opt
        self.class_num = opt.class_num
        self.confusion_matrix = torch.zeros((opt.class_num, opt.class_num), device=torch.device(opt.device))

    def update(self, predict, target):
        n = self.class_num
        with torch.no_grad():
            mask = (0<=target) & (target<n)
            value_p_t = n * target[mask].to(torch.int64) + predict[mask]
            self.confusion_matrix += torch.bincount(value_p_t, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        smooth=1e-3
        hist = self.confusion_matrix.cpu()

        tp = hist.diag()
        fp = hist.sum(dim=0) - hist.diag()
        fn = hist.sum(dim=1) - hist.diag()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * (precision * recall+smooth) / (precision + recall+smooth)).nanmean()

        pixel_acc = hist.trace() / hist.sum()
        class_pixel_acc = (hist.diag()+smooth) / (hist.sum(dim=1)+smooth)
        mean_pixel_class_acc = torch.nanmean(class_pixel_acc)

        iou = (hist.diag()+smooth) / (hist.sum(1) + hist.sum(0) - hist.diag()+smooth)
        miou = torch.nanmean(iou)
        freq = (hist.sum(dim=1)+smooth) / (hist.sum()+smooth)
        freq_weighted_iou=(freq[freq > 0] * iou[freq > 0]).sum()

        po = pixel_acc
        pe = (hist.sum(dim=0)/hist.sum() * hist.sum(dim=1)/hist.sum()).sum()
        kappa = (po - pe+smooth) / (1 - pe+smooth)

        return {'f1':f1.item(), 'pa':pixel_acc.item(),'cpa':class_pixel_acc.tolist(),'mpa':mean_pixel_class_acc.item(),
                'iou':iou.tolist(),'miou':miou.item(),'fwiou':freq_weighted_iou.item(),'kappa':kappa.item()}

    def reset(self):
        self.confusion_matrix = torch.zeros((self.opt.class_num, self.opt.class_num), device=torch.device(self.opt.device))



