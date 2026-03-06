import yaml
from easydict import EasyDict

def parallel_stateDict_to_normal_stateDict(state_dict):
    # 去除'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉 'module.' 前缀
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    return new_state_dict

def cfg_reader(cfg_path):
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    
    return config