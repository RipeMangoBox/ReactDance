import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
import os 
from  scipy.ndimage import gaussian_filter as G
from scipy import linalg
from scipy.signal import argrelextrema
import pandas as pd
import re

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPLX_FEAT_POINT = [0, 7, 8, 10, 11, 15, 16, 17, 20, 21]


SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', 
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', 
    'left_foot', # 10
    'right_foot', # 11
    'neck',
    'left_collar',
    'right_collar',
    'head', # 15
    'left_shoulder', # 16
    'right_shoulder', # 17
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #num_joints
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]

num_joints = 22
dim_joints = num_joints * 3

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def duet_feature(posef, posel):
    """
        posef: Tx55x3
        posel: Tx55x3
    """
    # Tx10x3
    Tf, _, _ = posef.shape
    Tl, _, _ = posel.shape
    T = np.min([Tf, Tl])
    posef = posef.copy()[:T, :]
    posel = posel.copy()[:T, :]
    # feat = np.sqrt(np.sum((posef[:, SMPLX_FEAT_POINT][:, :, None, :] - posel[:, SMPLX_FEAT_POINT][:, None, :, :])**2, axis=-1)).reshape(T, -1)
    feat = np.sqrt(np.sum((posef[:, SMPLX_FEAT_POINT][:, None, :, :] - posel[:, SMPLX_FEAT_POINT][:, :, None, :])**2, axis=-1)).reshape(T, -1) # 由于进行了reshape，融合了l和f被扩充的维度，因此两种扩充计算等价
    feat = np.mean(feat, axis=0)

    return feat

def compute_mpjpe(pred, target, scale=1000.0):
    """
    Compute MPJPE (Mean Per Joint Position Error) in millimeters.
    pred: [T, J, 3] predicted 3D joint coordinates
    target: [T, J, 3] ground truth 3D joint coordinates
    scale: unit conversion factor (e.g., meters to millimeters)
    """
    assert pred.shape == target.shape
    error = np.sqrt(np.sum((pred - target)**2, axis=-1))  # [T, J]
    return np.mean(error) * scale

def compute_mpjve(pred, target, scale=1000.0):
    """
    Compute standard MPJVE (Mean Per Joint Velocity Error) in millimeters (per frame).
    pred: [T, J, 3] predicted 3D joint coordinates (numpy array)
    target: [T, J, 3] ground truth 3D joint coordinates (numpy array)
    scale: unit conversion factor (e.g., meters to millimeters)
    """
    assert pred.shape == target.shape
    assert pred.ndim == 3 and pred.shape[-1] == 3
    # Compute displacement differences: [T-1, J, 3]
    pred_diff = pred[1:] - pred[:-1]
    target_diff = target[1:] - target[:-1]
    # L2 norm per joint per frame: [T-1, J]
    error = np.sqrt(np.sum((pred_diff - target_diff)**2, axis=-1))
    # Mean over all frames and joints
    return np.mean(error) * scale

def compute_jitter(pred, target, scale=1000.0):
    """
    Compute jitter (mean L2 distance of velocity differences) in millimeters.
    pred: [T, J, 3] predicted 3D joint coordinates
    target: [T, J, 3] ground truth 3D joint coordinates
    scale: scaling factor for jitter (default from reference code)
    """
    assert pred.shape == target.shape
    pred_vel = pred[1:] - pred[:-1]  # [T-1, J, 3]
    target_vel = target[1:] - target[:-1]  # [T-1, J, 3]
    error = np.sqrt(np.sum((pred_vel - target_vel)**2, axis=-1))  # [T-1, J]
    return np.mean(error) * scale

def compute_jitter_transition(pred, target, stride=240, half_interval=30, scale=1000.0):
    """
    Compute jitter within sliding windows centered at multiples of `stride`.
    
    Args:
        pred: [T, J, 3] predicted 3D joint coordinates
        target: [T, J, 3] ground truth 3D joint coordinates
        stride: interval between window centers (in frames)
        half_interval: half window size (so total window = 2 * half_interval + 1)
        scale: scaling factor (e.g., to convert to mm)

    Returns:
        mean_jitter: average jitter across all valid windows (scalar, in mm)
    """
    assert pred.shape == target.shape
    T = pred.shape[0]
    
    window_jitters = []
    window_size = 2 * half_interval + 1  # total frames in window

    # Iterate over window centers: 0, stride, 2*stride, ...
    for center in range(0, T, stride):
        start = center - half_interval
        end = center + half_interval + 1  # +1 for slicing

        # Skip if window goes out of bounds
        if start < 0 or end > T:
            continue

        pred_window = pred[start:end]      # [W, J, 3]
        target_window = target[start:end]  # [W, J, 3]

        # Compute velocity differences (W-1, J, 3)
        pred_vel = pred_window[1:] - pred_window[:-1]
        target_vel = target_window[1:] - target_window[:-1]

        # L2 error per joint per frame (W-1, J)
        error = np.sqrt(np.sum((pred_vel - target_vel) ** 2, axis=-1))

        # Mean over time and joints
        jitter = np.mean(error) * scale
        window_jitters.append(jitter)

    if not window_jitters:
        return 0.0  # or np.nan

    return np.mean(window_jitters)

def quantized_metrics(predicted_pkl_root, gt_pkl_root, no_divgt=False):
    pred_features = []
    gt_features = []
    pred_joints = []
    gt_joints = []

    # Load duet features and joint data
    pred_duet_dir = os.path.join(predicted_pkl_root, 'duet_features')
    gt_duet_dir = os.path.join(gt_pkl_root, 'duet_features')
    
    # Load duet features for FID
    pred_features = [np.load(os.path.join(pred_duet_dir, pkl)) for pkl in os.listdir(pred_duet_dir)]
    gt_features = [np.load(os.path.join(gt_duet_dir, pkl)) for pkl in os.listdir(gt_duet_dir)]
    pred_features = np.stack(pred_features)
    gt_features = np.stack(gt_features)

    # Load joint data for MPJPE and jitter
    pred_files = [pkl for pkl in os.listdir(predicted_pkl_root) if pkl.endswith('_00.npy')]
    # gt_files = [pkl for pkl in os.listdir(gt_pkl_root) if pkl.endswith('_00.npy')]
    for pklf in pred_files:
        joint3df = np.load(os.path.join(predicted_pkl_root, pklf))[..., :dim_joints].reshape(-1, num_joints, 3)
        joint3df = joint3df - joint3df[:, 0, None]
        pred_joints.append(joint3df)
        # time align
        gt_joint3df = np.load(os.path.join(gt_pkl_root, pklf))[:joint3df.shape[0], :dim_joints].reshape(-1, num_joints, 3)
        gt_joint3df = gt_joint3df - gt_joint3df[:, 0, None]
        gt_joints.append(gt_joint3df)


    # Normalize features for FID
    gt_features, pred_features = normalize(gt_features, pred_features)
    
    # Compute FID
    fid = calc_fid(pred_features, gt_features)
    
    # Compute diversity
    div = calculate_avg_distance(pred_features)
    div_gt = calculate_avg_distance(gt_features) if not no_divgt else None

    # Compute MPJPE and jitter for reactor (joint3df)
    mpjpe_f, mpjve_f, jitter_f, jitter_f_transition = [], [], [], []
    for pred_f, gt_f in zip(pred_joints, gt_joints):
        mpjpe_f.append(compute_mpjpe(pred_f, gt_f))
        mpjve_f.append(compute_mpjve(pred_f, gt_f))
        jitter_f.append(compute_jitter(pred_f, gt_f))
        jitter_f_transition.append(compute_jitter_transition(pred_f, gt_f))
    
    metrics = {
        'fid_k': fid,
        'div': div,
        'mpjpe_follower': np.mean(mpjpe_f),
        'mpjve_follower': np.mean(mpjve_f),
        'jitter_follower': np.mean(jitter_f),
        'jitter_transition': np.mean(jitter_f_transition),
    }
    
    if not no_divgt:
        metrics['div_gt'] = div_gt
        
    return metrics

def calc_fid(kps_gen, kps_gt):
    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats_duet(root):
    output_path = os.path.join(root, 'duet_features')
    if num_joints != 22:
        output_path = output_path + "_w_hand"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pkls = os.listdir(root)
    for pkl in pkls:
        if not pkl.endswith('_00.npy'):
            continue
        pkll = pkl.replace('_00.npy', '_01.npy')

        if os.path.isdir(os.path.join(root, pkl)):
            continue
        if not os.path.exists(os.path.join(root, pkll)):
            continue

        jointsf = np.load(os.path.join(root, pkl))[..., :dim_joints]
        jointsl = np.load(os.path.join(root, pkll))[..., :dim_joints]
        if len(jointsf.shape) == 3:
            jointsf = jointsf[0]
            jointsl = jointsl[0]
            
        n_joints = jointsf.shape[-1]//3
        
        # to fix the root translation in DD100 dataset
        rootl_init = jointsl[0, :3]
        jointsf = jointsf - np.tile(rootl_init, (jointsf.shape[0], n_joints))
        jointsl = jointsl - np.tile(rootl_init, (jointsl.shape[0], n_joints))
        
        joint3df = jointsf.reshape([-1, n_joints, 3])
        joint3dl = jointsl.reshape([-1, n_joints, 3])
        
        np.save(os.path.join(output_path, pkl), duet_feature(joint3df, joint3dl))

def calc_db(keypoints, name=''):
    n_joints = keypoints.shape[-1]//3
    keypoints = np.array(keypoints).reshape(-1, n_joints, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)

def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats[0]:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats[0]))

def calc_duet_be_score(root):
    ba_scores = []
    pkls = os.listdir(root)
    for pklf in pkls:
        if not pklf.endswith('_00.npy'):
            continue
        pkll = pklf.replace('_00.npy', '_01.npy')

        if os.path.isdir(os.path.join(root, pklf)):
            continue
        joint3df = np.load(os.path.join(root, pklf))
        joint3dl = np.load(os.path.join(root, pkll))
        dance_beatsf, lengthf = calc_db(joint3df)  
        dance_beatsl, lengthl = calc_db(joint3dl)     
        ba_scores.append(BA(dance_beatsl, dance_beatsf))
    return np.mean(ba_scores)

def get_roots_from_dirs(pred_dirs):
    """
    找到指定文件夹下所有包含 'npy' 或 'pos3d' 子文件夹的路径。

    参数:
        root_dir (str): 要搜索的根文件夹路径。

    返回:
        list: 包含符合条件的父文件夹路径列表。
    """
    result = []

    for pred_dir in pred_dirs:
        # 遍历根目录及其子目录
        for dirpath, dirnames, _ in os.walk(pred_dir):
            # 检查当前文件夹是否包含 'npy' 或 'pos3d' 子文件夹
            if 'pos3d_npy' in dirnames:
                result.append(os.path.join(dirpath, 'pos3d_npy'))  # 记录父文件夹路径

    return result

def shorten_path(pred_root, index):
    """
    Shorten a model path by removing specific prefix and suffix, and reformatting epoch.
    
    Args:
        pred_root (str): Original model path.
        index (int): Index for prefixing the shortened path.
    
    Returns:
        str: Shortened path with format 'index. path_with_epoch'.
    """
    shortened_path = pred_root
    # Remove prefix and suffix
    prefix = 'results/generated/ReactDance/'
    suffix = '/pos3d_npy'
    if shortened_path.startswith(prefix):
        shortened_path = shortened_path[len(prefix):]
    if shortened_path.endswith(suffix):
        shortened_path = shortened_path[:-len(suffix)]
    # Extract and reformat epoch
    epoch_match = re.search(r'epoch=(\d+)', shortened_path)
    if epoch_match:
        epoch = epoch_match.group(1)
        shortened_path = re.sub(r'fid_best_epoch=\d+-Eval_FID=[0-9.]+.ckpt', f'e{epoch}', shortened_path)
    return f"{index}. {shortened_path}"

def shorten_path_iclr_re(pred_root, index):
    return pred_root
    """
    Shorten a model path by removing specific prefix and suffix, and reformatting epoch.
    
    Args:
        pred_root (str): Original model path.
        index (int): Index for prefixing the shortened path.
    
    Returns:
        str: Shortened path with format 'index. path_with_epoch'.
    """
    shortened_path = pred_root
    # Remove prefix and suffix
    # prefix = 'results/generated/ReactDance/ICLR_re'
    # suffix = '/pos3d_npy'
    # if shortened_path.startswith(prefix):
    #     shortened_path = shortened_path[len(prefix):].split('/')[5]
    shortened_path = pred_root.split('/')[4]
    # Extract and reformat epoch
    epoch_match = re.search(r'epoch=(\d+)', shortened_path)
    if epoch_match:
        epoch = epoch_match.group(1)
        shortened_path = re.sub(r'fid_best_epoch=\d+-Eval_FID=[0-9.]+.ckpt', f'e{epoch}', shortened_path)
    return f"{index}. {shortened_path}"

def sort_metrics_df(df):
    """
    Sort DataFrame by FID (low), Diversity (high), MPJPE (low), Jitter (low), BE Score (high).
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'FID', 'Diversity', 'MPJPE (mm)', 'Jitter', 'BE Score'.
    
    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(
        by=['FID', 'Diversity', 'MPJPE (mm)', 'MPJVE (mm)', 'Jitter', 'BE Score'],
        ascending=[True, False, True, True, True, False]
        # by=['Jitter'],
        # ascending=[True]
    )
    
if __name__ == '__main__':
    gt_root = 'data_lazy/motion/pos3d/test'
    music_root = 'data_lazy/music/feature/test'

    pred_roots = [
        gt_root,
        'results/generated/HFSQ/hfsq128_512_q2/epoch_200.ckpt/samples/duet/pos3d_npy', # HFSQ recon
        'results/generated/ReactDance/reactdance/epoch_500.ckpt/samples/pos3d_npy', # ReactDance sampling
    ]

    # run this line for only once
    # calc_and_save_feats_duet(gt_root)
    # Collect results
    ranks = {}
    table_data = []
    for i, pred_root in enumerate(pred_roots):
        root_nb_path = shorten_path_iclr_re(pred_root, i)
        print(f"{'-'*15} {root_nb_path} {'-'*15}")
        calc_and_save_feats_duet(pred_root)
        be_score = calc_duet_be_score(pred_root)
        print(f"BE Score: {be_score:.4f}")
        qms = quantized_metrics(pred_root, gt_root, no_divgt=True)
        print(qms)
        ranks[root_nb_path] = qms
        # Prepare table row
        table_data.append({
            'Model Path': root_nb_path,
            'FID': qms['fid_k'],
            'Diversity': qms['div'],
            'MPJPE (mm)': qms['mpjpe_follower'],
            'MPJVE (mm)': qms['mpjve_follower'],
            'Jitter': qms['jitter_follower'],
            'Jitter_Transition': qms['jitter_transition'],
            'BE Score': be_score
        })
        print('\n')
    
    # Convert to pandas DataFrame and sort
    df = pd.DataFrame(table_data)
    df = sort_metrics_df(df)
    # Set display precision
    pd.set_option('display.float_format', '{:.4f}'.format)
    # Override precision for MPJPE (mm) to 2 decimal places
    # df['MPJPE (mm)'] = df['MPJPE (mm)'].map('{:.2f}'.format)
    # df['MPJVE (mm)'] = df['MPJVE (mm)'].map('{:.2f}'.format)
    print("\nEvaluation Results (Sorted by FID↓, Diversity↑, MPJPE↓, Jitter↓, BE Score↑):")
    print(df.to_string(index=False))
        
    
    # # Print sorted ranks by FID
    # print("\nSorted by FID:")
    # sorted_ranks = dict(sorted(ranks.items(), key=lambda item: item[1]['fid_k']))
    # for root_nb_path, metrics in sorted_ranks.items():
    #     print(f"{root_nb_path}: {metrics}")