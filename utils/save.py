import numpy as np
import os


def save_smplx(smplxf, translf, smplxl, transll, config, evaldir, epoch_tested, fname):
    if smplxf is None or smplxl is None:
        print('smplxf or smplxl is None, skippping smpl saving')
        return
    save_folder = os.path.join(evaldir, 'npy', 'ep'+str(epoch_tested).zfill(4))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    posesf = smplxf
    global_orientf = smplxf[:, 0]
    # translf = smplxf[:, :3]

    posesl = smplxl
    global_orientl = smplxl[:, 0]
    # transll = smplxl[:, :3]

    dictf = np.array([{'poses': posesf, 'global_orient': global_orientf, "transl": translf, 'meta': {'gender': 'female'}}])
    dictl = np.array([{'poses': posesl, 'global_orient': global_orientl, "transl": transll, 'meta': {'gender': 'male'}}])

    np.save(os.path.join(save_folder, fname + '_00'), dictf)
    np.save(os.path.join(save_folder, fname + '_01'), dictl)

def save_pos3d(posf, posl, evaldir, fname, suffix='pos3d_npy'):
    save_folder = os.path.join(evaldir, suffix)
    os.makedirs(save_folder, exist_ok=True)
    
    np.save(os.path.join(save_folder, fname + '_00'), posf)
    np.save(os.path.join(save_folder, fname + '_01'), posl)