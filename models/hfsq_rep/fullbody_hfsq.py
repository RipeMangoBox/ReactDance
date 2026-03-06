import numpy as np
import torch
import torch.nn as nn
import os

from .part_hfsq import Part_HFSQ
from models.model_utils.logger import average_metrics

SMPLX_JOINT_NAMES = [
    'pelvis',  # 0
    'left_hip',  # 1
    'right_hip',  # 2
    'spine1',  # 3
    'left_knee',  # 4
    'right_knee',  # 5
    'spine2',  # 6
    'left_ankle',  # 7
    'right_ankle',  # 8
    'spine3',  # 9
    'left_foot',  # 10
    'right_foot',  # 11
    'neck',  # 12
    'left_collar',  # 13
    'right_collar',  # 14
    'head',  # 15
    'left_shoulder',  # 16
    'right_shoulder',  # 17
    'left_elbow',  # 18
    'right_elbow',  # 19
    'left_wrist',  # 20
    'right_wrist',  # 21
    'jaw',  # 22
    'left_eye_smplhf',  # 23
    'right_eye_smplhf',  # 24
    'left_index1',  # 25
    'left_index2',  # 26
    'left_index3',  # 27
    'left_middle1',  # 28
    'left_middle2',  # 29
    'left_middle3',  # 30
    'left_pinky1',  # 31
    'left_pinky2',  # 32
    'left_pinky3',  # 33
    'left_ring1',  # 34
    'left_ring2',  # 35
    'left_ring3',  # 36
    'left_thumb1',  # 37
    'left_thumb2',  # 38
    'left_thumb3',  # 39
    'right_index1',  # 40
    'right_index2',  # 41
    'right_index3',  # 42
    'right_middle1',  # 43
    'right_middle2',  # 44
    'right_middle3',  # 45
    'right_pinky1',  # 46
    'right_pinky2',  # 47
    'right_pinky3',  # 48
    'right_ring1',  # 49
    'right_ring2',  # 50
    'right_ring3',  # 51
    'right_thumb1',  # 52
    'right_thumb2',  # 53
    'right_thumb3'  # 54
]

# which joint is not used in the kinematic chain?

kinematic_chain = [[0, 1, 4, 7, 10],    # left leg
                   [0, 2, 5, 8, 11],    # right leg
                   [0, 3, 6, 9, 12, 15],# spin
                   [9, 13, 16, 18, 20], # left arm  
                   [9, 14, 17, 19, 21], # right arm
                   
                   [15, 23],            # left eye
                   [15, 24],            # right eye
                   
                   [20, 25, 26, 27],    # left index
                   [20, 28, 29, 30],    # left middle
                   [20, 31, 32, 33],    # left pinky
                   [20, 34, 35, 36],    # left ring
                   [20, 37, 38, 39],    # left thumb
                   
                   [21, 40, 41, 42],    # right index
                   [21, 43, 44, 45],    # right middle
                   [21, 46, 47, 48],    # right pinky
                   [21, 49, 50, 51],    # right ring
                   [21, 52, 53, 54]    # right thumb
                ]

smpl_down = [0, 1, 2, 4,  5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
smpl_lhand = list(range(25, 40))
smpl_rhand = list(range(40, 55))

class FullBody_HFSQ(nn.Module):
    def __init__(self, hps, num_joints=22):
        super().__init__()

        self.hps = hps
        full_num_joints = 25 if num_joints == 22 else 55
        self.smpl_down = smpl_down
        self.smpl_up = smpl_up[:len(smpl_up) - (full_num_joints - num_joints)]
        self.n_joints = num_joints
        self.chanel_num = hps.joint_channel
        
        self.hfsq_up = Part_HFSQ(hps.up_half, hps.dim)
        self.hfsq_down = Part_HFSQ(hps.down_half, hps.dim)
        self.hfsq_tr = Part_HFSQ(hps.tr_half, hps.dim)

    def decode(self, zs, decode_mode='indices'):
        """
        Decode latent representations for reactor.
        zs: [z_up_follower, z_down_follower]
        decode_mode: 'indices' or 'embs'
        """
        zup, zdown, ztr = zs

        # Decode upper and lower body for reactor
        xup = self.hfsq_up.decode(zup, decode_mode=decode_mode)
        xdown = self.hfsq_down.decode(zdown, decode_mode=decode_mode)
        xtr = self.hfsq_tr.decode(ztr, decode_mode=decode_mode)
        
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        xup = xup.view(b, t, cup//self.chanel_num, self.chanel_num)
        xdown = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num)
                
        # Process reactor output
        xout = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num, dtype=xup.dtype, device=xup.device)
        xout[:, :, self.smpl_up] = xup
        xout[:, :, self.smpl_down] = xdown
        
        xout = xout.view(b, t, -1)
        xout[:, :, 0] = torch.zeros_like(xout[:, :, 0], dtype=xup.dtype, device=xup.device)
        x_out = {'pos3df': xout, 'translf': xtr}

        return x_out
    
    def rootl_full_forward(self, pos3dl):
        b, t, c = pos3dl.size()
        rootl = pos3dl[:, :, :3].clone()
        pos3dl = pos3dl.view(b, t, -1, 3) - rootl.unsqueeze(-2)
        return pos3dl.view(b, t, -1), rootl

    def encode(self, batch, encode_mode="all_quantizeds", ret_mode='reactor'):
        '''
        ret_mode: 'reactor', 'leader'
        '''
        pose_seqf, pose_seql = batch['pos3df'], batch['pos3dl']
        b, t, c = pose_seqf.size()
        
        pose_seq = pose_seqf if ret_mode == 'reactor' else pose_seql
        translf = pose_seqf[:, :, :3] - pose_seql[:, :, :3] if ret_mode == 'reactor' else pose_seql[:, :, :3] - pose_seqf[:, :, :3]
        pose_seq = pose_seq.view(b, t, -1, 3) - pose_seq[:, :, :3].unsqueeze(-2)

        xup = pose_seq[:, :, self.smpl_up, :].view(b, t, -1)
        xdown = pose_seq[:, :, self.smpl_down, :].view(b, t, -1)

        zup = self.hfsq_up.encode(xup, encode_mode=encode_mode)
        zdown = self.hfsq_down.encode(xdown, encode_mode=encode_mode)
        ztr = self.hfsq_tr.encode(translf, encode_mode=encode_mode)
        
        zout = (zup, zdown, ztr)
        
        return zout

    def _reconstruct_person(self, gt_pos_leader, gt_pos_follower):
        """
        Performs VQ-VAE reconstruction for a single person (the "reactor")
        relative to a reference person (the "leader").

        Args:
            gt_pos_leader (torch.Tensor): Ground truth pose of the leader. (B, T, C)
            gt_pos_follower (torch.Tensor): Ground truth pose of the reactor. (B, T, C)

        Returns:
            tuple: Contains:
                - pos_out (torch.Tensor): Reconstructed reactor pose.
                - transl_out (torch.Tensor): Reconstructed relative translation.
                - losses (dict): Dictionary of losses for this person.
                - metrics (dict): Dictionary of metrics for this person.
        """
        b, t, c = gt_pos_follower.size()
        losses = {}
        metrics = {}

        # --- 1. Data Preparation ---
        # Calculate relative translation from root positions
        transl = gt_pos_follower[:, :, :3] - gt_pos_leader[:, :, :3]
        
        # Center the reactor's pose by subtracting its own root position
        pose_seq = gt_pos_follower.view(b, t, -1, 3) - gt_pos_follower[:, :, :3].unsqueeze(-2)

        # Split into upper and lower body parts
        xup = pose_seq[:, :, self.smpl_up, :].view(b, t, -1)
        xdown = pose_seq[:, :, self.smpl_down, :].view(b, t, -1)
            
        # --- 2. VQ-VAE Forward Pass ---
        x_out_up, loss_up, metrics_up = self.hfsq_up(xup)
        x_out_down, loss_down, metrics_down = self.hfsq_down(xdown)
        x_out_tr, loss_tr, metrics_tr = self.hfsq_tr(transl)

        # --- 3. Reassemble the Pose ---
        cup = x_out_up.size(-1)
        cdown = x_out_down.size(-1)
        
        x_out_up = x_out_up.view(b, t, cup // self.chanel_num, self.chanel_num)
        x_out_down = x_out_down.view(b, t, cdown // self.chanel_num, self.chanel_num)

        # Initialize output tensor and fill with reconstructed parts
        pos_out = torch.zeros_like(pose_seq, dtype=x_out_up.dtype, device=x_out_up.device)
        pos_out[:, :, self.smpl_up] = x_out_up
        pos_out[:, :, self.smpl_down] = x_out_down
            
        # The root of the reconstructed pose is zero-centered
        pos_out[:, :, 0] = torch.zeros_like(pos_out[:, :, 0], dtype=x_out_up.dtype, device=x_out_up.device)
        transl_out = x_out_tr
        
        # --- 4. Collect Losses and Metrics ---
        losses['up_half'] = loss_up
        losses['down_half'] = loss_down
        losses['tr'] = loss_tr
        losses['total'] = loss_up + loss_down + loss_tr
        
        metrics.update({f'metrics_up/{k}': v for k, v in metrics_up.items()})
        metrics.update({f'metrics_down/{k}': v for k, v in metrics_down.items()})
        metrics.update({f'metrics_tr/{k}': v for k, v in metrics_tr.items()})
        return pos_out, transl_out, losses, metrics
    
    def forward(self, batch):
        """
        Unified VQ-VAE forward pass, handling single ('reactor') or dual ('duet')
        """
        pos3dl, pos3df = batch['pos3dl'], batch['pos3df']
        
        total_losses = {}
        total_metrics = {}

        # --- Mode 1: Reconstruct only the reactor (existing behavior) ---
        pos3df_out, translf_out, f_losses, f_metrics = self._reconstruct_person(
            gt_pos_leader=pos3dl, gt_pos_follower=pos3df
        )
        
        # Add 'follower_' prefix to all individual loss/metric keys
        for key, val in f_losses.items():
            total_losses[f'follower_{key}'] = val
        for key, val in f_metrics.items():
            total_metrics[f'follower_{key}'] = val

        # Set the main total loss for backpropagation
        total_losses['total'] = f_losses['total']
        
        # Organize output dictionary
        x_out = {'pos3dl': pos3dl, 'pos3df': pos3df_out, 'translf': translf_out}


        return x_out, total_losses, total_metrics
    
    # def forward(self, batch):
    #     """
    #     统一的 VQ-VAE 前向传播，处理双人的 pos 和 transl 数据。
        
    #     参数:
    #         batch: dict, 包含 'pos3dl', 'pos3df', 'translf'
        
    #     返回:
    #         x_out: dict, 包含 'pos3dl', 'pos3df', 'translf' 的输出
    #         losses: dict, 包含 pos 和 transl 的损失
    #         metrics: dict, 包含 pos 和 transl 的指标
    #     """
    #     pose_seql, pose_seqf = batch['pos3dl'], batch['pos3df']
    #     b, t, c = pose_seql.size()
    #     losses = {}
    #     metrics = {}
    #     person = 'reactor'

    #     translf = pose_seqf[:, :, :3] - pose_seql[:, :, :3]
    #     pose_seqf = pose_seqf.view(b, t, -1, 3) - pose_seqf[:, :, :3].unsqueeze(-2)
        
    #     # 分割上半身和下半身
    #     # pose_seqf = pose_seqf.view(b, t, c//self.chanel_num, self.chanel_num)
    #     xup = pose_seqf[:, :, self.smpl_up, :].view(b, t, -1)
    #     xdown = pose_seqf[:, :, self.smpl_down, :].view(b, t, -1)
        
    #     # VQ-VAE 前向传播
    #     x_out_up, loss_up, metrics_up = self.hfsq_up(xup)
    #     x_out_down, loss_down, metrics_down = self.hfsq_down(xdown)
    #     x_out_tr, loss_tr, metrics_tr = self.hfsq_tr(translf)
        
    #     # 合并输出
    #     cup = x_out_up.size(-1)
    #     cdown = x_out_down.size(-1)
        
    #     x_out_up = x_out_up.view(b, t, cup//self.chanel_num, self.chanel_num)
    #     x_out_down = x_out_down.view(b, t, cdown//self.chanel_num, self.chanel_num)
        
    #     xout = torch.zeros(b, t, (cup + cdown)//self.chanel_num, self.chanel_num, dtype=x_out_up.dtype, device=pose_seqf.device)
    #     xout[:, :, self.smpl_up] = x_out_up
    #     xout[:, :, self.smpl_down] = x_out_down

    #     xout_translf = x_out_tr
    #     xout[:, :, 0] = torch.zeros_like(xout_translf)

    #     # 存储损失和指标
    #     losses[f'{person}_up_half'] = loss_up
    #     losses[f'{person}_down_half'] = loss_down
    #     losses[f'{person}_tr'] = loss_tr
    #     losses[f'{person}_total'] = loss_up + loss_down + loss_tr
    #     metrics.update({
    #         f'{person}_metrics_up/{k}': v for k, v in metrics_up.items()
    #     })
    #     metrics.update({
    #         f'{person}_metrics_down/{k}': v for k, v in metrics_down.items()
    #     })
    #     metrics.update({
    #         f'{person}_metrics_tr/{k}': v for k, v in metrics_tr.items()
    #     })
        
    #     pos3df_out = xout

    #     total_loss = losses['follower_total']
    #     losses['total'] = total_loss
        
    #     # 组织输出
    #     x_out = {'pos3dl': pose_seql, 'pos3df': pos3df_out, 'translf': xout_translf}
        
    #     return x_out, losses, metrics
