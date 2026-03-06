import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os
import random
from utils.utils import rotate_motion_data, mirror_motion_data

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33], [20, 34, 35, 36], [20, 37, 38, 39]]
t2m_right_hand_chain = [[21, 40, 41, 42], [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51], [21, 52, 53, 54]]

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

def swap_left_right(data):
    def swap_left_right_position(data):
        data[..., 0] *= -1
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
        # left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
        # right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]
        left_hand_chain = [i for i in range(25, 40)]
        right_hand_chain = [i for i in range(40, 55)]

        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp
        if data.shape[1] > 24:
            tmp = data[:, right_hand_chain]
            data[:, right_hand_chain] = data[:, left_hand_chain]
            data[:, left_hand_chain] = tmp
        return data

    T, n_joints = data.shape[0], data.shape[1] // 3
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    positions = swap_left_right_position(positions).reshape(T, -1)

    return positions

def put_on_floor(positions):
    n = positions.shape[0]
    positions = positions.reshape(n, -1, 3)
    floor_height = positions.min(axis=0).min(axis=0)[2]  # z为高度
    positions[:, :, 2] -= floor_height
    return positions.reshape(n, -1)

class DD100lfAll(Dataset):
    '''
    we do not use rot6d in ReactDance training. If interested, you can have a try.
    '''
    def __init__(self, training_mode, music_root, motion_root, split='train', interval=None, dtype='pos3d', move=8, expansion=1, shuffle=False, num_joints=22, rotate_prob=0, mirror_prob=0, swap_prob=0):
        self.training_mode = training_mode
        self.music_opt = "feature"
        
        # Add both pos3d and rot6d if dtype is 'both'
        if dtype == 'both':
            dtypes = ['pos3d', 'rot6d']
        elif isinstance(dtype, list):
            dtypes = dtype
        else:
            dtypes = [dtype]
        self.dtype = dtype
        self.expansion = expansion
        self.names = []
        self.rotate_prob = rotate_prob
        self.mirror_prob = mirror_prob
        self.swap_prob = swap_prob
        self.shuffle = shuffle

        self.num_joints = num_joints
        self.dances = {'pos3dl':[], 'pos3df':[], 'rot6dl':[], 'rot6df':[], 'music':[]}
        music_files = {}
        agent_files = {'leader':{}, 'reactor':{}}

        def _process_files(agent_files, np_music):       
            def _process_dance(np_dance, dtype_folder):
                np_dance = np_dance[:, :np_dance.shape[1]//55*self.num_joints]
                if dtype_folder == 'pos3d':
                    np_dance = put_on_floor(np_dance)

                return np_dance

            this_pair = {}
            for agent in agent_files:
                # For each dtype_folder, load the corresponding file
                for dtype_folder in dtypes:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    if not os.path.isfile(dance_path):
                        continue
                    np_dance = np.load(dance_path)
                    np_dance = np_dance[:len(np_dance) - len(np_dance)%4, :] # to fit encodec down sample stragegy
                    np_dance = _process_dance(np_dance, dtype_folder)
                    this_pair[(agent, dtype_folder)] = np_dance

            # Only proceed if both leader and reactor exist for all dtypes
            for dtype_folder in dtypes:
                if (('leader', dtype_folder) not in this_pair) or (('reactor', dtype_folder) not in this_pair):
                    continue
                ldance = this_pair[('leader', dtype_folder)]
                fdance = this_pair[('reactor', dtype_folder)]
                
                # directly move the lroot to origin and froot correspond to it
                if dtype_folder == 'pos3d':
                    lroot = ldance[0, :3]
                    fdance = fdance - np.tile(lroot, (fdance.shape[0], self.num_joints))
                    ldance = ldance - np.tile(lroot, (ldance.shape[0], self.num_joints))
                
                lenf, dim = ldance.shape
                lenl, dim = fdance.shape
                seq_len = min(lenf, lenl)

                if (interval is not None) and ( interval != 'None' ):
                    for i in range(0, seq_len - interval, move):
                        np_dance_sub_seq_l = ldance[i: i + interval]
                        np_dance_sub_seq_f = fdance[i: i + interval]
                        np_music_sub_seq = np_music[i: i + interval]

                        if len(np_dance_sub_seq_l) != interval or len(np_dance_sub_seq_f) != interval or len(np_music_sub_seq) != interval:
                            continue

                        self.dances[dtype_folder+'l'].append(np_dance_sub_seq_l)
                        self.dances[dtype_folder+'f'].append(np_dance_sub_seq_f)
                        if dtype_folder == dtypes[0]:
                            self.dances['music'].append(np_music_sub_seq)
                            self.names.append(take)
                else:
                    self.dances[dtype_folder+'l'].append(ldance[:seq_len//move*move])
                    self.dances[dtype_folder+'f'].append(fdance[:seq_len//move*move])
                    if dtype_folder == dtypes[0]:
                        self.dances['music'].append(np_music[:seq_len//move*move])
                        self.names.append(take)

        fnames = os.listdir(os.path.join(motion_root, 'pos3d', split))
        mnames = os.listdir(os.path.join(music_root,  self.music_opt, split))
        if split == 'all':
            fnames = os.listdir(os.path.join(motion_root, 'pos3d', 'train')) + os.listdir(os.path.join(motion_root, 'pos3d', 'test'))
            mnames = os.listdir(os.path.join(music_root,  self.music_opt, 'train')) + os.listdir(os.path.join(music_root,  self.music_opt, 'test'))
       
        for mname in mnames:
            path = os.path.join(music_root, self.music_opt, split, mname)
            music_files[mname[:-4]] = path

        for fname in fnames:
            path = os.path.join(motion_root, 'pos3d', split, fname)
            if path.endswith('_00.npy'):
                agent_files['reactor'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path

        for take in agent_files['reactor']:
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            # For each dtype, process files
            _process_files(agent_files, np_music)

        # align the length of pos3d and music
        for i in range(len(self.dances['pos3dl'])):
            min_len = min(len(self.dances['pos3dl'][i]), len(self.dances['pos3df'][i]), len(self.dances['rot6dl'][i]), len(self.dances['rot6df'][i]), len(self.dances['music'][i]))
            min_len = min_len // 4 * 4
            self.dances['pos3dl'][i] = self.dances['pos3dl'][i][:min_len]
            self.dances['pos3df'][i] = self.dances['pos3df'][i][:min_len]
            self.dances['music'][i] = self.dances['music'][i][:min_len]
            if self.dances['rot6dl'][i] is not None:
                self.dances['rot6dl'][i] = self.dances['rot6dl'][i][:min_len]
            if self.dances['rot6df'][i] is not None:
                self.dances['rot6df'][i] = self.dances['rot6df'][i][:min_len]

        # 创建统一的索引列表
        dataset_size = len(self.dances['pos3dl'])
        self.shuffle_indices = list(range(dataset_size))
        # 在初始化时打乱索引
        random.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.dances['pos3dl'])*self.expansion

    def __getitem__(self, index):
        index = index // self.expansion
        if self.shuffle:
            index = self.shuffle_indices[index]
        pos3dl = self.dances['pos3dl'][index]
        pos3df = self.dances['pos3df'][index]
        # Add rot6d data if available
        rot6dl = self.dances['rot6dl'][index] if len(self.dances['rot6dl']) > 0 else None
        rot6df = self.dances['rot6df'][index] if len(self.dances['rot6df']) > 0 else None
        
        if np.random.rand() < self.rotate_prob:
            # select from center
            rotate_center = 'root' if np.random.rand() < 0.5 else 'centroid'
            angle = np.random.randint(0, 15)
            axis = 'z'
            pos3df, rot6df = rotate_motion_data(pos3df, rot6df, angle_deg=angle, axis=axis, center=rotate_center)

        if np.random.rand() < self.mirror_prob:
            mirror_center = 'root' if np.random.rand() < 0.5 else 'centroid'
            pos3df, rot6df = mirror_motion_data(pos3df, rot6df, mirror_center=mirror_center)
            pos3dl, rot6dl = mirror_motion_data(pos3dl, rot6dl, mirror_center=mirror_center)

        if np.random.rand() < self.swap_prob:
            pos3dl, pos3df = pos3df, pos3dl
            if rot6dl is not None and rot6df is not None:
                rot6dl, rot6df = rot6df, rot6dl
        
        item = {
            'pos3dl':pos3dl, 
            'pos3df':pos3df, 
            'music':self.dances['music'][index], 
            'fname':self.names[index]
        }
        # if rot6dl is not None and rot6df is not None:
        #     item['rot6dl'] = rot6dl
        #     item['rot6df'] = rot6df

        return item
