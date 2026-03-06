import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import os 
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import json
from tqdm import tqdm
import pandas as pd
import re

def normalize(feat, feat2):
    mean1, mean2 = feat.mean(axis=0), feat2.mean(axis=0)
    std1, std2 = feat.std(axis=0), feat2.std(axis=0)
    return (feat - mean1)/(std1 + 1e-10), (feat2 - mean2)/(std2 + 1e-10)

num_joints = 22
xyz_joints = num_joints * 3
eps = 1e-5

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
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
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


def quantized_metrics(predicted_pkl_root, gt_pkl_root):
    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]
    
    pred_features_k = np.stack(pred_features_k)[:, :xyz_joints]  # Nx72 p40
    pred_features_m = np.stack(pred_features_m)[:, :xyz_joints] # Nx32

    gt_freatures_k = np.stack(gt_freatures_k)[:, :xyz_joints] # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m)[:, :xyz_joints] # 
    
    print("Number of predicted sequences:", pred_features_m.shape[0])
    print("Number of ground truth sequences:", gt_freatures_m.shape[0])
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # 
    std_gt = gt_freatures_m.std(axis=0)
    mask = std_gt > 1e-5
    gt_freatures_m_reduced = gt_freatures_m[:, mask]
    pred_features_m_reduced = pred_features_m[:, mask]
    fid_g = calc_fid(pred_features_m_reduced, gt_freatures_m_reduced)
    
    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_g = calc_fid(pred_features_m, gt_freatures_m)
    # fid_g = calc_fid(pred_features_m_reduced, gt_freatures_m_reduced)
    
    # check
    # std_gt = gt_freatures_m.std(axis=0)
    # low_variance_indices = np.where(std_gt < 1e-5)[0]
    # print("Low variance features (std < 1e-5):", low_variance_indices)
    # print("Corresponding std values:", std_gt[low_variance_indices])
    
    
    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_g_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_g = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_g': fid_g, 'div_k': div_k, 'div_g' : div_g}
    return metrics


def calc_fid(kps_gen, kps_gt):
    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False) + eps * np.eye(kps_gen.shape[1])
    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False) + eps * np.eye(kps_gt.shape[1])
    ...

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    # Product might be almost singular

    covprod = sigma_gen.dot(sigma_gt)
    covmean = linalg.sqrtm(covprod)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
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

def calc_and_save_feats(root):
    os.makedirs(os.path.join(root, 'kinetic_features'), exist_ok=True)
    os.makedirs(os.path.join(root, 'manual_features_new'), exist_ok=True)
    pred_list = []

    for pkl in tqdm(os.listdir(root), desc='Extracting features'):
        if not pkl.endswith('_00.npy'):
            continue

        if os.path.isdir(os.path.join(root, pkl)):
            continue

        joint3d = np.load(os.path.join(root, pkl))[..., :xyz_joints]
        # squeeze the batch dimension
        if len(joint3d.shape) == 3:
            joint3d = joint3d[0]

        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        joint3d = joint3d - np.tile(roott, (1, num_joints))  # Calculate relative offset with respect to root

        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, num_joints, 3)))
        np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, num_joints, 3)))


def get_mb(music_root, key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        # sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.load(path)[:, 53][:length]
        else:
            beats = np.load(path)[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]

        return beat_axis


def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, num_joints, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
   
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):

    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def calc_ba_score(root, music_root):

    # gt_list = []
    
    ba_scores = []

    for pkl in os.listdir(root):
        if not pkl.endswith('_00.npy'):
            continue

        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl))[..., :xyz_joints]
        
        if len(joint3d.shape) == 3:
            joint3d = joint3d[0]
        len_motion = len(joint3d)
        len_music = len(np.load(os.path.join(music_root, pkl.split('_')[0] + '_' + pkl.split('_')[1] + '_' + pkl.split('_')[2] + '.npy')))
        len_min = min(len_motion, len_music)
        joint3d = joint3d[:len_min]


        dance_beats, length = calc_db(joint3d, pkl) 
        music_beats = get_mb(music_root, pkl.split('_')[0] + '_' + pkl.split('_')[1] + '_' + pkl.split('_')[2] + '.npy', length)
        ba_scores.append(BA(   music_beats, dance_beats[0],))
        # joint3d = None

        
    return np.mean(ba_scores)

def sort_metrics_df(df):
    """
    Sorts the DataFrame based on specified metric priorities.
    
    Args:
        df (pd.DataFrame): The DataFrame containing evaluation metrics.
    
    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(
        by=['fid_k', 'fid_g', 'div_k', 'div_g', 'BA Score'],
        ascending=[True, True, False, False, False]
    )


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

if __name__ == '__main__':


    gt_root = 'data_lazy/motion/pos3d/test'
    music_root = 'data_lazy/music/feature/test'

    pred_roots = [
        gt_root,
        'results/generated/HFSQ/hfsq128_512_q2/epoch_200.ckpt/samples/duet/pos3d_npy', # HFSQ recon
        'results/generated/ReactDance/reactdance/epoch_500.ckpt/samples/pos3d_npy', # ReactDance sampling
    ]
        
    table_data = []
    print("--- Calculating Prediction Metrics ---")
    for i, pred_root in enumerate(tqdm(pred_roots, desc="Evaluating Models")):
        # Pre-calculate features for the current prediction
        calc_and_save_feats(pred_root)
        
        # Calculate metrics
        ba_score = calc_ba_score(pred_root, music_root)
        qms = quantized_metrics(pred_root, gt_root)
        
        print(f"BA Score: {ba_score:.4f}, Metrics: {qms}")
        
        # Append data for the table
        row = {
            'Path': pred_root,
            'BA Score': ba_score,
            'fid_k': qms.get('fid_k'),
            'fid_g': qms.get('fid_g'),
            'div_k': qms.get('div_k'),
            'div_g': qms.get('div_g')
        }
        table_data.append(row)

    # --- Display Sorted Results ---
    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort the DataFrame according to the desired metric priorities
    sorted_df = sort_metrics_df(df)
    
    # Set display options for better readability
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_colwidth', None)
    
    print("\n" + "="*50)
    print("      Sorted Evaluation Results      ")
    print("="*50)
    print("Sorted by: FID_k↓, fid_g↓, Div_k↑, div_g↑, BA Score↑")
    print(sorted_df.to_string(index=False))
    print("="*50)