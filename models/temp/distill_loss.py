#from yolov6.models.losses.loss_distill import ComputeLoss as ComputeLoss_distill

#ComputeLoss_distill()

import torch
import torch.nn.functional as F


"""distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01- 1) + 1
d_loss_dfl *= distill_weightdecay
d_loss_cls *= distill_weightdecay
d_loss_cw *= distill_weightdecay"""


"""loss_weight={
        'class': 1.0,
        'iou': 2.5,
        'dfl': 0.5,
        'cwd': 10.0},

distill_weight={
        'class': 1.0,
        'dfl': 1.0,
        }"""

def _df_loss(self, pred_dist, target):
        
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

def distill_loss_dfl(logits_student, logits_teacher,reg_max,temperature=20):

        logits_student = logits_student.view(-1,reg_max)
        logits_teacher = logits_teacher.view(-1,reg_max)
        pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        

        d_loss_dfl = F.kl_div(pred_student, pred_teacher, reduction="none")
        d_loss_dfl = d_loss_dfl.sum(1).mean()
        d_loss_dfl *= temperature**2

        log_pred_student = torch.log(pred_student)
        #d_loss_dfl = F.mse_loss(logits_student, logits_teacher, reduction="mean")

        return d_loss_dfl

def distill_loss_dfl_2(logits_student, logits_teacher,reg_max,temperature=20):

        logits_student = logits_student.view(-1,reg_max)
        logits_teacher = logits_teacher.view(-1,reg_max)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature**2
        return d_loss_dfl


def distill_loss_cls_2(logits_student, logits_teacher, num_classes, temperature=20):

        logits_student = logits_student.view(-1, num_classes)
        logits_teacher = logits_teacher.view(-1, num_classes)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_cls = F.kl_div(log_pred_student, pred_teacher,reduce="batchmean")                #reduction="sum")
        d_loss_cls *= temperature**2

        return d_loss_cls


def distill_loss_cls(logits_student, logits_teacher, num_classes, temperature=20):

        logits_student = logits_student.view(-1, num_classes)
        logits_teacher = logits_teacher.view(-1, num_classes)
        pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        #log_pred_student = torch.log(pred_student)

        d_loss_cls = F.kl_div(pred_student, pred_teacher, reduction="batchmean")
        d_loss_cls *= temperature**2

        #d_loss_cls = F.mse_loss(logits_student, logits_teacher, reduction="mean") 
        return d_loss_cls


def distill_loss_cw(s_feats, t_feats,  temperature=1):
        N,C,H,W = s_feats[0].shape
        # print(N,C,H,W)
        loss_cw = F.kl_div(F.log_softmax(s_feats[0].view(N,C,H*W)/temperature, dim=2),
                           F.log_softmax(t_feats[0].view(N,C,H*W).detach()/temperature, dim=2),
                           reduction='sum',
                           log_target=True) * (temperature * temperature)/ (N*C)

        N,C,H,W = s_feats[1].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[1].view(N,C,H*W)/temperature, dim=2),
                           F.log_softmax(t_feats[1].view(N,C,H*W).detach()/temperature, dim=2),
                           reduction='sum',
                           log_target=True) * (temperature * temperature)/ (N*C)

        N,C,H,W = s_feats[2].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[2].view(N,C,H*W)/temperature, dim=2),
                           F.log_softmax(t_feats[2].view(N,C,H*W).detach()/temperature, dim=2),
                           reduction='sum',
                           log_target=True) * (temperature * temperature)/ (N*C)
        # print(loss_cw)
        return loss_cw

def distillation_wasserstein_loss(s_feats, t_feats):
    """
    Computes Wasserstein (Earth Mover's Distance) loss for ANN-to-SNN distillation
    across multiple feature maps.

    Args:
        s_feats (list of tensors): List of (N, C, H, W) feature maps from the student (SNN).
        t_feats (list of tensors): List of (N, C, H, W) feature maps from the teacher (ANN).
    
    Returns:
        torch.Tensor: Wasserstein loss value.
    """
    loss = 0
    num_layers = len(s_feats)  # Number of feature maps

    for s_feat, t_feat in zip(s_feats, t_feats):  # Iterate through all feature maps
        N, C, H, W = s_feat.shape

        # Flatten feature maps to (N, C, H*W) for spatial comparison
        s_feat = s_feat.view(N, C, H * W)
        t_feat = t_feat.view(N, C, H * W).detach()  # Detach teacher to avoid backprop through it

        # Sort each channel-wise feature map
        s_sorted, _ = torch.sort(s_feat, dim=-1)  # Sort across spatial dimension
        t_sorted, _ = torch.sort(t_feat, dim=-1)  # Sort across spatial dimension

        # Compute Wasserstein-1 Distance (Mean Absolute Difference)
        loss += torch.mean(torch.abs(s_sorted - t_sorted))  # Wasserstein Distance

    return loss / num_layers  # Normalize by the number of feature maps

def distillation_mse_loss(s_feats,t_feats):
       
       mse_list = [F.mse_loss(a, b.detach()) for a, b in zip(s_feats, t_feats)]

       return torch.sum(torch.stack(mse_list))


def distill_loss_feature_maps(s_feature_maps,t_feature_maps,from_snn_distill):

        s_feature_maps_Treduced = [torch.mean(fmap,dim=0) for fmap in s_feature_maps]

        if from_snn_distill:
                t_feature_maps_Treduced = [torch.mean(fmap,dim=0) for fmap in t_feature_maps]
        else:
                t_feature_maps_Treduced = t_feature_maps 

        return distillation_mse_loss(s_feature_maps_Treduced, t_feature_maps_Treduced)