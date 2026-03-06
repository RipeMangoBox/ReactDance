import os
import numpy as np
import torch
# import cv2
from PIL import Image
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import savgol_filter
from models.model_utils.quaternion import qbetween_np, qrot_np

face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints, separate=False):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    if not separate:
        new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    else:
        new_data = positions.reshape(T, -1), rotations.reshape(T, -1)
    return new_data

trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])

def process_dd100lf_motion_np(motion, feet_thre, prev_frames, n_joints):
    # (seq_len, joints_num, 3), motion=(seq_len, joints_num*3+rot_num)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]
    up_axis = 2 # z up (now z is height)
    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    # 不再变换坐标轴，假设输入已经是xy平面，z为高度
    # positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)  # 不再需要

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[2]  # z为高度
    positions[:, :, 2] -= floor_height

    '''XY at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])  # 只保留xy，z为0
    positions = positions - root_pose_init_xy

    '''All initially face Y+ (正y轴)'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around z-axis
    forward_init = np.cross(across, np.array([[0, 0, 1]]), axis=-1)  # z轴朝上，右手定则
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 1, 0]])  # y轴正方向
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 2]  # z为高度
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 2]  # z为高度
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1] # [frame, 66=(joints)22*3] 0+66=66
    data = np.concatenate([data, joint_vels], axis=-1) # [frame, 66=(joints)22*6], 66+66=132
    data = np.concatenate([data, rot_data[:-1]], axis=-1) # [frame, 126=21*6], 132+126=258
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # both feet are [frame, 2], 258+2+2=262

    return data, root_quat_init, root_pose_init_xy[None]


import numpy as np
from scipy.spatial.transform import Rotation

def rot6d_to_rotmat(rot6d):
    """
    将 6D 旋转表示转换为旋转矩阵。
    
    参数:
        rot6d: np.ndarray, 形状为 (..., 6)，6D 旋转向量
    返回:
        rotmat: np.ndarray, 形状为 (..., 3, 3)，旋转矩阵
    """
    b, n = rot6d.shape[:2]
    rot6d = rot6d.reshape(b*n, 6)
    x = rot6d[..., :3]
    y = rot6d[..., 3:]
    
    # 正交化：Gram-Schmidt 过程
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    
    rotmat = np.stack([x, y, z], axis=-1)  # 形状: (..., 3, 3)
    return rotmat.reshape(b, n, 3, 3)

def rotmat_to_rot6d(rotmat):
    """
    将旋转矩阵转换为 6D 旋转表示。
    
    参数:
        rotmat: np.ndarray, 形状为 (..., 3, 3)，旋转矩阵
    返回:
        rot6d: np.ndarray, 形状为 (..., 6)，6D 旋转向量
    """
    rot6d = rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6)  # 取前两列
    return rot6d

def rotate_motion_data(pos3d_data, rot6d_data=None, angle_deg=0.0, axis='z', center='centroid'):
    """
    对 3D 位置数据和（可选的）6D 旋转表示进行旋转增强。
    
    参数:
        pos3d_data: np.ndarray, 形状为 (n, nj*3)，3D 关节位置数据（不可为 None）
        rot6d_data: np.ndarray, 形状为 (n, nj*6)，6D 旋转数据（可为 None）
        angle_deg: float, 旋转角度（度）
        axis: str, 旋转轴，可选 'x', 'y', 'z'
        center: str, 旋转中心，可选 'centroid'（质心）或 'root'（根关节，pos3d[:, :3]）
    
    返回:
        rotated_rot6d: np.ndarray 或 None，旋转后的 6D 旋转数据（如果 rot6d_data 为 None，则返回 None）
        rotated_pos3d: np.ndarray, 旋转后的 3D 位置数据
    """
    # 确保 pos3d_data 不为 None 且形状正确
    if pos3d_data is None:
        raise ValueError("pos3d_data 不可为 None")
    n, d_pos = pos3d_data.shape
    assert d_pos % 3 == 0, "pos3d_data 的第二维必须是 3 的倍数"
    nj = d_pos // 3
    
    # 如果 rot6d_data 不为 None，检查形状
    if rot6d_data is not None:
        n_rot, d_rot = rot6d_data.shape
        assert d_rot % 6 == 0, "rot6d_data 的第二维必须是 6 的倍数"
        assert n == n_rot, "时间帧数必须一致"
    
    # 重塑数据
    pos3d_reshaped = pos3d_data.reshape(n, nj, 3)  # (n, nj, 3)
    rot6d_reshaped = rot6d_data.reshape(n, nj, 6) if rot6d_data is not None else None
    
    # 计算旋转中心
    if center == 'centroid':
        rotation_center = np.mean(pos3d_reshaped, axis=1, keepdims=True)  # 形状: (n, 1, 3)
    elif center == 'root':
        rotation_center = pos3d_reshaped[:, :1, :]  # 根关节，形状: (n, 1, 3)
    else:
        raise ValueError("center 必须是 'centroid' 或 'root'")
    
    # 将 pos3d 数据平移到旋转中心
    pos3d_centered = pos3d_reshaped - rotation_center
    
    # 将角度从度转换为弧度
    angle_rad = np.deg2rad(angle_deg)
    
    # 定义旋转矩阵
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=np.float32)
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError("轴必须是 'x', 'y' 或 'z'")
    
    # 旋转 pos3d 数据
    rotated_pos3d = np.einsum('ij,nkj->nki', rotation_matrix, pos3d_centered)
    rotated_pos3d = rotated_pos3d + rotation_center  # 平移回原中心
    
    # 旋转 rot6d 数据（如果存在）
    rotated_rot6d = None
    if rot6d_data is not None:
        rotmat = rot6d_to_rotmat(rot6d_reshaped)  # 转换为旋转矩阵，形状: (n, nj, 3, 3)
        rotated_rotmat = np.einsum('ij,nkjl->nkil', rotation_matrix, rotmat)  # 应用全局旋转
        rotated_rot6d = rotmat_to_rot6d(rotated_rotmat)  # 转换回 6D 表示
        rotated_rot6d = rotated_rot6d.reshape(n, nj*6)
    
    # 重塑 pos3d 数据回原始形状
    rotated_pos3d = rotated_pos3d.reshape(n, nj*3)
    
    return rotated_pos3d, rotated_rot6d

def mirror_motion_data(pos3d_data, rot6d_data=None, mirror_axis='x', mirror_center='centroid'):
    """
    对 3D 位置数据和（可选的）6D 旋转数据进行镜像处理（默认沿 y-z 平面）。
    
    参数:
        pos3d_data: np.ndarray, 形状为 (n, nj*3)，3D 关节位置数据（不可为 None）
        rot6d_data: np.ndarray, 形状为 (n, nj*6)，6D 旋转数据（可为 None）
        mirror_axis: str, 镜像轴，可选 'x', 'y', 'z'（默认沿 x 轴镜像，即 y-z 平面）
        mirror_center: str, 镜像中心，可选 'centroid'（质心）或 'root'（根关节，pos3d[:, :3]）
    
    返回:
        mirrored_pos3d: np.ndarray, 镜像后的 3D 位置数据
        mirrored_rot6d: np.ndarray 或 None，镜像后的 6D 旋转数据（如果 rot6d_data 为 None，则返回 None）
    """
    # 确保 pos3d_data 不为 None 且形状正确
    if pos3d_data is None:
        raise ValueError("pos3d_data 不可为 None")
    n, d_pos = pos3d_data.shape
    assert d_pos % 3 == 0, "pos3d_data 的第二维必须是 3 的倍数"
    nj = d_pos // 3
    
    # 如果 rot6d_data 不为 None，检查形状
    if rot6d_data is not None:
        n_rot, d_rot = rot6d_data.shape
        assert d_rot % 6 == 0, "rot6d_data 的第二维必须是 6 的倍数"
        assert n == n_rot, "时间帧数必须一致"
    
    # 重塑数据
    pos3d_reshaped = pos3d_data.reshape(n, nj, 3)  # (n, nj, 3)
    rot6d_reshaped = rot6d_data.reshape(n, nj, 6) if rot6d_data is not None else None
    
    # 计算镜像中心
    if mirror_center == 'centroid':
        mirror_center_pos = np.mean(pos3d_reshaped, axis=1, keepdims=True)  # 形状: (n, 1, 3)
    elif mirror_center == 'root':
        mirror_center_pos = pos3d_reshaped[:, :1, :]  # 根关节，形状: (n, 1, 3)
    else:
        raise ValueError("mirror_center 必须是 'centroid' 或 'root'")
    
    # 将 pos3d 数据平移到镜像中心
    pos3d_centered = pos3d_reshaped - mirror_center_pos
    
    # 定义镜像矩阵
    if mirror_axis == 'x':
        mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # 翻转 x 轴
    elif mirror_axis == 'y':
        mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)  # 翻转 y 轴
    elif mirror_axis == 'z':
        mirror_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)  # 翻转 z 轴
    else:
        raise ValueError("镜像轴必须是 'x', 'y' 或 'z'")
    
    # 镜像 pos3d 数据
    mirrored_pos3d = np.einsum('ij,nkj->nki', mirror_matrix, pos3d_centered)
    mirrored_pos3d = mirrored_pos3d + mirror_center_pos  # 平移回原中心
    
    # 镜像 rot6d 数据（如果存在）
    mirrored_rot6d = None
    if rot6d_data is not None:
        rotmat = rot6d_to_rotmat(rot6d_reshaped)  # 转换为旋转矩阵
        mirrored_rotmat = np.einsum('ij,nkjl->nkil', mirror_matrix, rotmat)  # 应用镜像变换
        mirrored_rot6d = rotmat_to_rot6d(mirrored_rotmat)  # 转换回 6D 表示
        mirrored_rot6d = mirrored_rot6d.reshape(n, nj*6)
    
    # 重塑 pos3d 数据回原始形状
    mirrored_pos3d = mirrored_pos3d.reshape(n, nj*3)
    
    return mirrored_pos3d, mirrored_rot6d

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%9fe-4' % (epoch, niter_state, inner_iter, lr*10000), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)

    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

def motion_temporal_filter(motion, filter_type="gaussian", filter_kwargs={"sigma": 1}):
    """
    对 motion 数据进行时间维度上的平滑处理。
    
    参数:
        motion (np.ndarray): 输入的 motion 数据，形状为 (b, n, d)，其中
                             b 是 batch_size，n 是时序长度，d 是关节位置维度。
        filter_type (str): 平滑滤波器类型，支持 "gaussian" 和 "savgol_filter"。
        sigma (float): 高斯滤波的标准差（仅在 filter_type="gaussian" 时使用）。
    
    返回:
        np.ndarray: 平滑后的 motion 数据，形状与输入相同。
    """
    b, n, d = motion.shape  # 获取 batch_size、时序长度和维度
    
    ret_tensor = torch.is_tensor(motion)
    if ret_tensor:
        device = motion.device
        motion = motion.cpu().numpy()
    
    if filter_type == "gaussian":
        # 使用高斯滤波对时间维度进行平滑
        smoothed_motion = gaussian_filter1d(motion, sigma=filter_kwargs["sigma"], axis=1, mode='nearest')
    
    elif filter_type == "savgol":
        # 使用 Savitzky-Golay 滤波对时间维度进行平滑
        # 注意：savgol_filter 不支持直接对多维数组操作，因此需要逐 batch 处理
        window_length = filter_kwargs["window_length"]  # 窗口长度
        polyorder = filter_kwargs["polyorder"]      # 多项式阶数
        
        # 初始化输出数组
        smoothed_motion = np.zeros_like(motion)
        
        for i in range(b):  # 对每个 batch 分别处理
            smoothed_motion[i] = savgol_filter(motion[i], window_length=window_length, polyorder=polyorder, axis=0)
    
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    
    if ret_tensor:
        smoothed_motion = torch.from_numpy(smoothed_motion).to(device)
    return smoothed_motion


if __name__ == '__main__':
    b, n, d = 7, 240, 66
    motion = np.random.randn(b, n, d)
    motion_gaussian = motion_temporal_filter(motion, "gaussian", 1)
    motion_savgol = motion_temporal_filter(motion, "savgol")
    
    print(motion_gaussian)
    print(motion_savgol)
    debug = 1