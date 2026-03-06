from torch import nn
import torch
import torch.nn.functional as F
from models.model_utils.quaternion import qbetween

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

forward_joint_indx = [2,1,17,16]   # right hip, left hip, right shoulder, left shoulder

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
 
class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Latent_'
        self.losses = {}
            
    def forward(self, latent_loss, train_mode='train'):
        if train_mode == 'train':
            self.losses[self.name + 'latent'] = torch.mean(latent_loss)
            self.accum_loss()
            
    def accum_loss(self):
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        self.losses.pop('latent_loss_total', 0)
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses['latent_loss_total'] = loss  

class GeometricLoss(nn.Module):
    def __init__(self, loss_fn='l1', n_joints=55, loss_checklist=None):
        super().__init__()
        if loss_fn == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif loss_fn == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif loss_fn == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.name = 'Geometry_'
        self.n_joints = n_joints
        self.fids = [7, 10, 8, 11]
        
        # 定义权重项及其默认值
        loss_weights_config = {
            'recon': 0.0,
            'vel': 0.0,
            'acc': 0.0,
            'rot6d_recon': 0.0,
            'rot6d_vel': 0.0,
            'rot6d_acc': 0.0,
            'FC': 0.0,
            'FL': 0.0,
            'BL': 0.0,
            'RO': 0.0,
            'DM': 0.0,
            'JA': 0.0,
            'TR_recon': 0.0,
            'TR_vel': 0.0,
            'TR_acc': 0.0,
        }

        # 初始化 weights 字典
        self.weights = {}

        for key, default_value in loss_weights_config.items():
            attr_name = f"w_{key}"
            self.weights[key] = getattr(loss_checklist, attr_name, default_value)

        print(f"GeometricLoss weight: {self.weights}")

        # 全局几何 loss 开关
        self.use_GEO = getattr(loss_checklist, "use_GEO", True)

        self.DM_thresh = 1
        self.JA_thresh = 0.1

        thres = 0.001
        self.velfactor = torch.Tensor([thres, thres, thres, thres]).to("cuda")
        self.heightfactor = torch.Tensor([0.12, 0.05, 0.12, 0.05]).to("cuda")
        self.losses = {}
        self.metrics = {}
        self.use_ROT6D_RECON = self._enabled("rot6d_recon")

    def _enabled(self, key: str) -> bool:
        """
        统一判断某个 loss 分支是否启用：
        只要对应权重非 0 即认为启用。
        """
        return self.weights.get(key, 0.0) != 0.0

    def _recon_vel_acc(self, pred, tar):
        """
        统一的重建 / 速度 / 加速度三项损失计算，支持任意形状 [..., C] 序列。
        """
        # reconstruction loss
        recon_loss = self.mix_masked_mse(pred, tar)

        # velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        tar_vel = tar[:, 1:] - tar[:, :-1]
        vel_loss = self.mix_masked_mse(pred_vel, tar_vel)

        # acceleration loss
        pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        tar_acc = tar[:, 2:] - 2 * tar[:, 1:-1] + tar[:, :-2]
        acc_loss = self.mix_masked_mse(pred_acc, tar_acc)

        return recon_loss, vel_loss, acc_loss

    def mix_masked_mse(self, prediction, target, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)

        return loss.mean()

    def R2G(self, pose_sample, pose_seql, translf_sample):
        pose_sample = pose_sample.clone()
        pose_sample[:, :, 0, :] = 0
        root_offset = pose_seql[:, :, :1, :] + translf_sample.unsqueeze(-2)
        pose_sample = pose_sample + root_offset

        return pose_sample, pose_seql

    def forward(self, motion_pred, motion_target, train_mode='train'):
        B, T, J = motion_pred['pos3df'].shape

        self.pred_posf = motion_pred['pos3df'].view(B, T, self.n_joints, 3)
        self.tar_posf = motion_target['pos3df'].view(B, T, self.n_joints, 3)
        self.global_gt_posl = motion_target['pos3dl'].view(B, T, self.n_joints, 3)
        self.global_pred_posf, _ = self.R2G(self.pred_posf, self.global_gt_posl, motion_pred['translf'])
        self.global_tar_posf, _ = self.R2G(self.tar_posf, self.global_gt_posl, motion_target['translf'])

        if train_mode == 'train':
            self.forward_reconstruction()
            self.forward_translation()
            if self._enabled("rot6d_recon") and 'rotf' in motion_pred and 'rotf' in motion_target:
                self.pred_rotf = motion_pred['rotf'].view(B, T, self.n_joints, 6)
                self.tar_rotf = motion_target['rotf'].view(B, T, self.n_joints, 6)
                self.forward_rot6d_reconstruction()
            self.forward_regularization()
            if self._enabled("RO"):
                self.forward_relative_rot()
            if self._enabled("DM") or self._enabled("JA"):
                self.forward_DM_JA(self.DM_thresh, self.JA_thresh)
            if self._enabled("BL"):
                self.forward_bone_length()
            if self._enabled("FC"):
                self.forward_contact()
            if self._enabled("FL"):
                self.forward_floating()
                               
        if not self.use_GEO:
            self.metrics.update(self.losses)
            self.losses = {}
        self.accum_loss()
    
    def forward_reconstruction(self):
        pred_posf = self.pred_posf
        tar_posf = self.tar_posf
        pos_loss, pos_vel_loss, pos_acc_loss = self._recon_vel_acc(pred_posf, tar_posf)

        if self._enabled("recon"):
            self.losses[self.name + 'recon'] = pos_loss
            self.losses[self.name + 'vel'] = pos_vel_loss
            self.losses[self.name + 'acc'] = pos_acc_loss
        else:
            self.metrics[self.name + 'recon'] = pos_loss
            self.metrics[self.name + 'vel'] = pos_vel_loss
            self.metrics[self.name + 'acc'] = pos_acc_loss

    def forward_translation(self):
        pred_transl = self.global_pred_posf[:, :, 0, :]
        tar_transl = self.global_tar_posf[:, :, 0, :]
        transl_loss, transl_vel_loss, transl_acc_loss = self._recon_vel_acc(
            pred_transl, tar_transl
        )

        if self._enabled("TR_recon"):
            self.losses[self.name + "TR_recon"] = transl_loss
            self.losses[self.name + "TR_vel"] = transl_vel_loss
            self.losses[self.name + "TR_acc"] = transl_acc_loss
        else:
            self.metrics[self.name + "TR_recon"] = transl_loss
            self.metrics[self.name + "TR_vel"] = transl_vel_loss
            self.metrics[self.name + "TR_acc"] = transl_acc_loss

    def forward_floating(self):
        # 1. 获取脚部 Z 轴高度 (绝对值)
        # shape: [Batch, Time, 4] (假设是 L_Ankle, L_Toe, R_Ankle, R_Toe)
        feet_h = torch.abs(self.global_pred_posf[:, :-1, self.fids, 2])
        
        # 2. 计算相对高度差 (Relative Difference)
        # 先让高度减去各自对应的阈值
        # 确保 heightfactor 维度匹配，变为 [1, 1, 4] 以便广播
        # relative_diff > 0 表示该关节悬空，< 0 表示该关节在阈值范围内
        relative_diff = feet_h - self.heightfactor
        
        # 3. 寻找相对位置最低的关节 (Find the "most grounded" joint)
        # 在关节维度 (dim=-1) 取 min。
        # 只要有一个关节的 relative_diff < 0 (即踩在地上)，min_relative 就会小于 0
        min_relative_h, _ = torch.min(relative_diff, dim=-1, keepdim=True)
        
        # 4. 计算 Loss
        # 如果最接近地面的关节都在阈值之上 (min_relative_h > 0)，则产生 Loss
        # 如果有任一关节在阈值内 (min_relative_h < 0)，Clamp 后 Loss 为 0
        loss_floating = torch.clamp(min_relative_h, min=0)
        
        # 5. 调用原本的 mse wrapper 进行 mask 处理和 reduction
        self.losses[self.name + "FL"] = self.mix_masked_mse(loss_floating, torch.zeros_like(loss_floating))
        
    def forward_contact(self):
        feet_vel = self.global_pred_posf[:, 1:, self.fids, :] - self.global_pred_posf[:, :-1, self.fids,:]
        feet_h = self.global_pred_posf[:, :-1, self.fids, 2]   # z-axis up, different from Intergen
        # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

        contact = self.foot_detect(feet_vel, feet_h)

        self.losses[self.name + "FC"] = self.mix_masked_mse(feet_vel, torch.zeros_like(feet_vel), contact)
        
    def foot_detect(self, feet_vel, feet_h):
        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2

        contact = (((feet_x + feet_y + feet_z) < self.velfactor) & (feet_h < self.heightfactor)).float()
        return contact
    
    def forward_regularization(self):
        pred_posf = self.pred_posf
        
        with torch.no_grad():
            pos_reg_loss = torch.mean(pred_posf ** 2)
            self.metrics['pos_regularization'] = pos_reg_loss
        
    def forward_bone_length(self):
        pred_posf = self.global_pred_posf
        tar_posf = self.global_tar_posf
        pred_bones = []
        tar_bones = []
        
        body_range = 17 if self.n_joints == 55 else 5
        for chain in kinematic_chain[:body_range]:
            for i, joint in enumerate(chain[:-1]):
                pred_bone = (pred_posf[..., chain[i], :] - pred_posf[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B,T,P,1]
                tar_bone = (tar_posf[..., chain[i], :] - tar_posf[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)
                pred_bones.append(pred_bone)
                tar_bones.append(tar_bone)

        pred_bones = torch.cat(pred_bones, dim=-1)
        tar_bones = torch.cat(tar_bones, dim=-1)

        self.losses[self.name + 'BL'] = self.mix_masked_mse(pred_bones, tar_bones)
        
    def forward_relative_rot(self):
        r_hip, l_hip, sdr_r, sdr_l = forward_joint_indx
        # Joint indices for root and head
        root_idx = 0
        neck_idx = 12
        head_idx = 15

        # For root forward direction (using hips for across, neck-root for up)
        # Follower predicted
        pos_root_f = self.global_pred_posf[..., root_idx, :]
        l_hip_f = self.global_pred_posf[..., l_hip, :]
        r_hip_f = self.global_pred_posf[..., r_hip, :]
        neck_f = self.global_pred_posf[..., neck_idx, :]
        acrossf = r_hip_f - l_hip_f
        acrossf_norm = acrossf.norm(dim=-1, keepdim=True)
        acrossf = acrossf / torch.clamp(acrossf_norm, min=1e-6)
        up_f = neck_f - pos_root_f
        up_norm = up_f.norm(dim=-1, keepdim=True)
        forwardf = torch.cross(up_f, acrossf, dim=-1)
        forwardf_norm = forwardf.norm(dim=-1, keepdim=True)
        forwardf = forwardf / torch.clamp(forwardf_norm, min=1e-6)

        # Follower target
        pos_root_f_tar = self.global_tar_posf[..., root_idx, :]
        l_hip_f_tar = self.global_tar_posf[..., l_hip, :]
        r_hip_f_tar = self.global_tar_posf[..., r_hip, :]
        neck_f_tar = self.global_tar_posf[..., neck_idx, :]
        acrossf_tar = r_hip_f_tar - l_hip_f_tar
        acrossf_tar_norm = acrossf_tar.norm(dim=-1, keepdim=True)
        acrossf_tar = acrossf_tar / torch.clamp(acrossf_tar_norm, min=1e-6)
        up_f_tar = neck_f_tar - pos_root_f_tar
        up_norm_tar = up_f_tar.norm(dim=-1, keepdim=True)
        forwardf_tar = torch.cross(up_f_tar, acrossf_tar, dim=-1)
        forwardf_tar_norm = forwardf_tar.norm(dim=-1, keepdim=True)
        forwardf_tar = forwardf_tar / torch.clamp(forwardf_tar_norm, min=1e-6)

        # Leader ground truth
        pos_root_l_gt = self.global_gt_posl[..., root_idx, :]
        l_hip_l_gt = self.global_gt_posl[..., l_hip, :]
        r_hip_l_gt = self.global_gt_posl[..., r_hip, :]
        neck_l_gt = self.global_gt_posl[..., neck_idx, :]
        acrossl_gt = r_hip_l_gt - l_hip_l_gt
        acrossl_gt_norm = acrossl_gt.norm(dim=-1, keepdim=True)
        acrossl_gt = acrossl_gt / torch.clamp(acrossl_gt_norm, min=1e-6)
        up_l_gt = neck_l_gt - pos_root_l_gt
        up_norm_l_gt = up_l_gt.norm(dim=-1, keepdim=True)
        forwardl_gt = torch.cross(up_l_gt, acrossl_gt, dim=-1)
        forwardl_gt_norm = forwardl_gt.norm(dim=-1, keepdim=True)
        forwardl_gt = forwardl_gt / torch.clamp(forwardl_gt_norm, min=1e-6)

        # Relative rotations for root
        pred_relative_rot_root = qbetween(forwardf, forwardl_gt)
        tar_relative_rot_root = qbetween(forwardf_tar, forwardl_gt)

        # Ignore nan values
        nan_mask_root = torch.isnan(pred_relative_rot_root) | torch.isnan(tar_relative_rot_root)
        pred_relative_rot_root[nan_mask_root] = 0.0
        tar_relative_rot_root[nan_mask_root] = 0.0

        # For head forward direction (using shoulders for across, head-neck for up)
        # Follower predicted
        head_f = self.global_pred_posf[..., head_idx, :]
        l_sdr_f = self.global_pred_posf[..., sdr_l, :]
        r_sdr_f = self.global_pred_posf[..., sdr_r, :]
        neck_f_head = self.global_pred_posf[..., neck_idx, :]  # reuse neck
        acrossf_head = r_sdr_f - l_sdr_f
        acrossf_head_norm = acrossf_head.norm(dim=-1, keepdim=True)
        acrossf_head = acrossf_head / torch.clamp(acrossf_head_norm, min=1e-6)
        up_f_head = head_f - neck_f_head
        up_norm_f_head = up_f_head.norm(dim=-1, keepdim=True)
        forwardf_head = torch.cross(up_f_head, acrossf_head, dim=-1)
        forwardf_head_norm = forwardf_head.norm(dim=-1, keepdim=True)
        forwardf_head = forwardf_head / torch.clamp(forwardf_head_norm, min=1e-6)

        # Follower target
        head_f_tar = self.global_tar_posf[..., head_idx, :]
        l_sdr_f_tar = self.global_tar_posf[..., sdr_l, :]
        r_sdr_f_tar = self.global_tar_posf[..., sdr_r, :]
        neck_f_tar_head = self.global_tar_posf[..., neck_idx, :]
        acrossf_tar_head = r_sdr_f_tar - l_sdr_f_tar
        acrossf_tar_head_norm = acrossf_tar_head.norm(dim=-1, keepdim=True)
        acrossf_tar_head = acrossf_tar_head / torch.clamp(acrossf_tar_head_norm, min=1e-6)
        up_f_tar_head = head_f_tar - neck_f_tar_head
        up_norm_tar_head = up_f_tar_head.norm(dim=-1, keepdim=True)
        forwardf_tar_head = torch.cross(up_f_tar_head, acrossf_tar_head, dim=-1)
        forwardf_tar_head_norm = forwardf_tar_head.norm(dim=-1, keepdim=True)
        forwardf_tar_head = forwardf_tar_head / torch.clamp(forwardf_tar_head_norm, min=1e-6)

        # Leader ground truth
        head_l_gt = self.global_gt_posl[..., head_idx, :]
        l_sdr_l_gt = self.global_gt_posl[..., sdr_l, :]
        r_sdr_l_gt = self.global_gt_posl[..., sdr_r, :]
        neck_l_gt_head = self.global_gt_posl[..., neck_idx, :]
        acrossl_gt_head = r_sdr_l_gt - l_sdr_l_gt
        acrossl_gt_head_norm = acrossl_gt_head.norm(dim=-1, keepdim=True)
        acrossl_gt_head = acrossl_gt_head / torch.clamp(acrossl_gt_head_norm, min=1e-6)
        up_l_gt_head = head_l_gt - neck_l_gt_head
        up_norm_l_gt_head = up_l_gt_head.norm(dim=-1, keepdim=True)
        forwardl_gt_head = torch.cross(up_l_gt_head, acrossl_gt_head, dim=-1)
        forwardl_gt_head_norm = forwardl_gt_head.norm(dim=-1, keepdim=True)
        forwardl_gt_head = forwardl_gt_head / torch.clamp(forwardl_gt_head_norm, min=1e-6)

        # Relative rotations for head
        pred_relative_rot_head = qbetween(forwardf_head, forwardl_gt_head)
        tar_relative_rot_head = qbetween(forwardf_tar_head, forwardl_gt_head)

        # Create a combined NaN mask for head: ignore if either pred or tar has NaN
        nan_mask_head = torch.isnan(pred_relative_rot_head) | torch.isnan(tar_relative_rot_head)
        pred_relative_rot_head[nan_mask_head] = 0.0
        tar_relative_rot_head[nan_mask_head] = 0.0

        # Additional losses for head-root orientation difference and sudden changes (on predicted reactor)

        # Head-Root consistency loss (penalize large differences > 60 degrees)
        dot_hr = torch.sum(forwardf * forwardf_head, dim=-1)
        cos_hr = torch.clamp(dot_hr, -1.0, 1.0)
        cos_thresh_consistency = torch.cos(torch.tensor(torch.pi / 3.0).to(cos_hr.device))  # cos(60°)
        consistency_loss = torch.mean(F.relu(cos_thresh_consistency - cos_hr))

        # Temporal smoothness losses (penalize sudden changes > 60 degrees per frame)
        smooth_loss_root = torch.tensor(0.0).to(forwardf.device)
        smooth_loss_head = torch.tensor(0.0).to(forwardf.device)
        if forwardf.size(1) > 1:
            # For root
            forward_curr_root = forwardf[:, 1:, :]
            forward_prev_root = forwardf[:, :-1, :]
            dot_delta_root = torch.sum(forward_curr_root * forward_prev_root, dim=-1)
            cos_delta_root = torch.clamp(dot_delta_root, -1.0, 1.0)
            cos_thresh_smooth = torch.cos(torch.tensor(torch.pi / 3.0).to(cos_delta_root.device))  # cos(30°)
            smooth_loss_root = torch.mean(F.relu(cos_thresh_smooth - cos_delta_root))

            # For head
            forward_curr_head = forwardf_head[:, 1:, :]
            forward_prev_head = forwardf_head[:, :-1, :]
            dot_delta_head = torch.sum(forward_curr_head * forward_prev_head, dim=-1)
            cos_delta_head = torch.clamp(dot_delta_head, -1.0, 1.0)
            smooth_loss_head = torch.mean(F.relu(cos_thresh_smooth - cos_delta_head))

        temporal_smooth_loss = (smooth_loss_root + smooth_loss_head) / 2

        # Losses
        RO_loss_root = self.mix_masked_mse(pred_relative_rot_root[..., [0, 2]], tar_relative_rot_root[..., [0, 2]])
        RO_loss_head = self.mix_masked_mse(pred_relative_rot_head[..., [0, 2]], tar_relative_rot_head[..., [0, 2]])
        # 在 losses 中分别记录
        self.metrics["Inter_RO_root"] = RO_loss_root
        self.metrics["Inter_RO_head"] = RO_loss_head
        self.metrics["Inter_RO_consistency"] = consistency_loss
        self.metrics["Inter_RO_temporal_smooth"] = temporal_smooth_loss
        self.losses["Inter_RO"] = RO_loss_root + RO_loss_head + consistency_loss + temporal_smooth_loss

    def forward_DM_JA(self, DM_thresh, JA_thresh):
        # 重塑关节点张量
        B, N = self.global_pred_posf.shape[:2]
        pred_posf = self.global_pred_posf.view(-1, self.n_joints, 3)
        tar_posf = self.global_tar_posf.view(-1, self.n_joints, 3)
        global_gt_posl = self.global_gt_posl.view(-1, self.n_joints, 3)
        
        # 计算关节点对之间的欧几里得距离
        pred_distance_matrix = torch.cdist(pred_posf, global_gt_posl).view(B, N, -1)
        tar_distance_matrix = torch.cdist(tar_posf, global_gt_posl).view(B, N, -1)
        
        def distance_loss(pred_distance_matrix, tar_distance_matrix, distance_matrix_mask):
            loss = self.mix_masked_mse(pred_distance_matrix, tar_distance_matrix, dm_mask=distance_matrix_mask)
            return loss

        # 记录损失
        if self._enabled("DM"):
            DM_distance_matrix_mask = (pred_distance_matrix < DM_thresh).float()
            self.losses["Inter_DM"] = distance_loss(
                pred_distance_matrix, tar_distance_matrix, DM_distance_matrix_mask
            )
        if self._enabled("JA"):
            JA_distance_matrix_mask = (tar_distance_matrix < JA_thresh).float()
            self.losses["Inter_JA"] = distance_loss(
                pred_distance_matrix,
                torch.zeros_like(tar_distance_matrix),
                JA_distance_matrix_mask,
            )
                    
    def forward_rot6d_reconstruction(self):
        pred_rot6d = self.pred_rotf
        tar_rot6d = self.tar_rotf
        rot_loss, rot_vel_loss, rot_acc_loss = self._recon_vel_acc(
            pred_rot6d, tar_rot6d
        )
        self.losses[self.name + "rot6d_recon"] = rot_loss
        self.losses[self.name + "rot6d_vel"] = rot_vel_loss
        self.losses[self.name + "rot6d_acc"] = rot_acc_loss
        
    def accum_loss(self):
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        self.losses.pop('geometry_loss_total', 0)
        for term in self.losses.keys():
            loss += self.losses[term] * self.weights[term.split('_')[-1]]
        self.losses['geometry_loss_total'] = loss
