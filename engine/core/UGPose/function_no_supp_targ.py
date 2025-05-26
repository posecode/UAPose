#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter
from .loss import DiscriminativeLoss
# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import get_final_preds, dark_get_final_preds
from datasets.transforms import reverse_transforms
from ..evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im
from einops import rearrange
from tabulate import tabulate

def cnt_grad(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm().item()

    return total_norm
import torch

def kl_div_log_sum_exp(prob, target):
    epsilon = 0.0000001  # 定义一个小的epsilon值
    prob = torch.clamp(prob, min = epsilon)  # 将prob的值限制在epsilon之上
    target = torch.clamp(target, min = epsilon)  # 将target的值限制在epsilon之上
    log_prob = torch.log(prob)
    log_target = torch.log(target)
    M = torch.max(log_prob, log_target)
    kl_div_sum = torch.sum(prob * (log_prob - M - (log_target - M)))
    exp_log_prob_minus_M = torch.exp(log_prob - M)
    exp_log_target_minus_M = torch.exp(log_target - M)
    log_ratio_sum = torch.log(torch.sum(exp_log_prob_minus_M) / torch.sum(exp_log_target_minus_M))
    return kl_div_sum + log_ratio_sum


import torch


import torch
import torch

def _test(model,frame):
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c , h , w = 1, 3 * frame, 256 , 192
    random_x = torch.randn((b, c , h  , w )).to('cuda')

    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")

def mmd_loss(N, U, sigma=1.0):
    batch_size_N, *other_dims_N = N.shape
    batch_size_U, *other_dims_U = U.shape
    N = N.view(batch_size_N, -1)
    U = U.view(batch_size_U, -1)

    # 计算高斯核函数
    def gaussian_kernel(x, y):
        pairwise_distance_squared = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
        return torch.exp(-pairwise_distance_squared / (2 * sigma ** 2))

    # 计算MMD
    k_NN = gaussian_kernel(N, N)
    k_UU = gaussian_kernel(U, U)
    k_NU = gaussian_kernel(N, U)

    mmd_value = torch.mean(k_NN) + torch.mean(k_UU) - 2 * torch.mean(k_NU)
    return mmd_value

def batch_wise_mse_loss(tensor1, tensor2):
    batchsize = tensor1.shape[0]
    mse_loss = torch.mean((tensor1 - tensor2) ** 2, dim=(1, 2, 3))
    return torch.mean(mse_loss)


def cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1.flatten(1), tensor2.flatten(1))

@CORE_FUNCTION_REGISTRY.register()
class Function_No_Supp_Targ_UGPose(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "UGPOSE")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()
        self.discriminative_loss = DiscriminativeLoss(usegpu=True).cuda()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        grad = AverageMeter()
        klloss = AverageMeter()
        kllossu = AverageMeter()
        kllossn = AverageMeter()
        lossm = AverageMeter()
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()
        if epoch in [0, 1, 2]:
            _test(model, self.cfg.MODEL.FRAME_NUM)


        for iter_step in range(self.max_iter_num):
            # if self.cfg.MODEL.FRAME_NUM == 3:
            #     input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight ,  meta = next(self.dataloader_iter)
            # else:
            input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)


            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            # if self.cfg.MODEL.FRAME_NUM == 3:
            #     concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
            # else:
            concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()

            outputs , htms_backbone , prob , prob_N , prob_U , N, U= model(concat_input)




            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps = outputs[0]
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                for pred_heatmaps in outputs[1:]:
                    loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                if self.cfg.MODEL.FREEZE_BACKBONE_WEIGHTS == False:
                    loss+=self.criterion(htms_backbone , target_heatmaps, target_heatmaps_weight)

            version = self.cfg.MODEL.VERSION
            # version = 'a'
            # version = 'b'
            # version = 'c'
            if version == 'a':
                kl_loss_x = 0
                kl_loss_U = 0
                kl_loss_N = 0
                loss_M = 0


            if version == 'b':
                prob = rearrange(prob, 'b j h w -> (b j) (h w)')
                # prob = F.softmax(prob / temperature , dim=-1)
                target = rearrange(target_heatmaps, 'b j h w -> (b j) (h w)')
                # target = F.softmax(target / temperature , dim=-1)

                kl_loss_x = F.kl_div(prob, target, reduction='mean')

                kl_loss_U = 0
                kl_loss_N = 0
                loss_M = 0

            if version == 'c':
                target = rearrange(target_heatmaps, 'b j h w -> (b j) (h w)')
                prob_N = rearrange(prob_N, 'b j h w -> (b j) (h w)')
                prob_U = rearrange(prob_U, 'b j h w -> (b j) (h w)')

                kl_loss_N = F.kl_div(prob_N, target, reduction='mean')
                kl_loss_U = F.kl_div(prob_U, target, reduction='mean')

                # N = rearrange(N, 'b j h w -> (b j) (h w)')
                # U = rearrange(U, 'b j h w -> (b j) (h w)')

                loss_M = (torch.mean(cosine_similarity(N, U)) + 1 + 0.0000001) / 2.0

                kl_loss_x = 0



            if version == 'full':
                prob = rearrange(prob , 'b j h w -> (b j) (h w)' )
                # prob = F.softmax(prob / temperature , dim=-1)
                target = rearrange(target_heatmaps , 'b j h w -> (b j) (h w)' )
                # target = F.softmax(target / temperature , dim=-1)


                kl_loss_x = F.kl_div(prob, target , reduction='mean')

                prob_N = rearrange(prob_N, 'b j h w -> (b j) (h w)')
                prob_U = rearrange(prob_U, 'b j h w -> (b j) (h w)')


                kl_loss_N = F.kl_div(prob_N, target, reduction='mean')
                kl_loss_U = F.kl_div(prob_U, target, reduction='mean')

                # N = rearrange(N, 'b j h w -> (b j) (h w)')
                # U = rearrange(U, 'b j h w -> (b j) (h w)')

                loss_M = (torch.mean(cosine_similarity(N, U))+1+0.0000001) / 2.0
                # loss_M = -1 * mmd_loss(N, U, sigma=1)
                # loss_M = kl_div_log_sum_exp(N, U) / (N.shape[0])



            temporal_loss =   kl_loss_U  - 0.1 *   kl_loss_N + 0.01 * loss_M


            loss = loss + 0.1 * kl_loss_x + 0.1 * temporal_loss


            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            grad.update(cnt_grad(model))
            optimizer.step()

            # measure accuracy and record loss
            if isinstance(kl_loss_x, torch.Tensor):
                klloss.update(kl_loss_x.item(), input_x.size(0))
            else:
                klloss.update(kl_loss_x, input_x.size(0))  # 假设kl_loss_x已经是标量

            def safe_update(loss_obj, loss_value, batch_size):
                """安全更新损失值的工具函数"""
                if isinstance(loss_value, torch.Tensor):
                    loss_obj.update(loss_value.item(), batch_size)
                elif isinstance(loss_value, (int, float)):
                    # print(f"Warning: loss_value is {type(loss_value)}. Expected Tensor.")
                    loss_obj.update(float(loss_value), batch_size)
                else:
                    raise TypeError(f"Unsupported loss type: {type(loss_value)}")

            # 修改后的调用方式
            safe_update(kllossu, kl_loss_U, input_x.size(0))
            safe_update(kllossn, kl_loss_N, input_x.size(0))
            safe_update(lossm, loss_M, input_x.size(0))

            losses.update(loss.item(), input_x.size(0))
            # kllossu.update(kl_loss_U.item(), input_x.size(0))
            # kllossn.update(kl_loss_N.item(), input_x.size(0))
            # lossm.update(loss_M.item(), input_x.size(0))

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'KL Loss x {kl_loss.val:.5f} ({kl_loss.avg:.5f})\t' \
                      'KL Loss u {kl_lossu.val:.5f} ({kl_lossu.avg:.5f})\t' \
                      'KL Loss n {kl_lossn.val:.5f} ({kl_lossn.avg:.5f})\t' \
                      'Loss m {lossm.val:.5f} ({lossm.avg:.5f})\t' \
                      'Grad {grad.val:.3f} ({grad.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num, batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, grad = grad,loss=losses, acc=acc , kl_loss = klloss,
                                                                        kl_lossu = kllossu, kl_lossn = kllossn, lossm = lossm)

                logger.info(msg)

            # For Tensorboard
            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):

        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc ,acc_backbone = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()


        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_preds_backbone = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_boxes = np.zeros((num_samples, 6))
        all_boxes_backbone = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        idx_backbone = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                # if self.cfg.MODEL.FRAME_NUM == 3:
                #     input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta = next(
                #         self.dataloader_iter)
                # else:
                #     input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = next(
                #         self.dataloader_iter)

                input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = next(
                    self.dataloader_iter)

                # input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight,  meta = next(self.dataloader_iter)

                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)

                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # if self.cfg.MODEL.FRAME_NUM == 3:
                #     concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
                # else:
                #     concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()

                concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()


                outputs , htms_backbone , _ , _ , _ , _, _  = model(concat_input.cuda())





                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, input_sup_A, input_sup_B])

                # from utils.vis import vis_vremd
                # vis_vremd(model.htms_dict, image=[input_x, input_sup_A, input_sup_B], target=target_heatmaps
                #          , batch_id=iter_step)

                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = outputs[-1]
                else:
                    pred_heatmaps = outputs

                FLIP_TEST = self.cfg.VAL.FLIP_VAL
                if FLIP_TEST:
                    concat_input_flipped = concat_input.flip(3)

                    pred_heatmaps_flipped, htms_backbone_flipped, _ ,_ ,_ ,_ , _ = model(
                        concat_input_flipped.cuda())
                    from datasets.process.pose_process import flip_back
                    pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(),
                                                      dataset.flip_pairs)
                    htms_backbone_flipped = flip_back(htms_backbone_flipped.cpu().numpy(), dataset.flip_pairs)

                    pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                    htms_backbone_flipped = torch.from_numpy(htms_backbone_flipped.copy()).cuda()

                    pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :,
                                                         0:-1]
                    htms_backbone_flipped[:, :, :, 1:] = htms_backbone_flipped.clone()[:, :, :, 0:-1]
                    pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                    htms_backbone = (htms_backbone + htms_backbone_flipped) * 0.5


                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                _, avg_acc_backbone, cnt_backbone, _ = accuracy(htms_backbone.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())

                acc_backbone.update(avg_acc_backbone, cnt_backbone)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t   \
                          {acc_backbone.val:.3f} ({acc_backbone.avg:.3f})'.format(phase, iter_step, num_batch, batch_time=batch_time,
                                                                          data_time=data_time, acc=acc, acc_backbone=acc_backbone)
                    logger.info(msg)



                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                preds, maxvals = dark_get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                preds_backbone, maxvals_backbone = dark_get_final_preds(htms_backbone.clone().cpu().numpy(), center, scale)
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 0:2] = preds_backbone[:, :, 0:2]
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 2:3] = maxvals_backbone
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 0:2] = center[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 2:4] = scale[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 5] = score
                idx_backbone += num_images


                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)


        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        name_values_backbone, perf_indicator_backbone = dataset.evaluate(self.cfg, all_preds_backbone, result_output_dir, all_boxes_backbone,
                                                       filenames_map, filenames, imgnums)

        self._print_name_value(name_values_backbone, 'ViT_Backbone')

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)), nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
                                                                      0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from ..vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir, "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir
