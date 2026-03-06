import sys
sys.path.append(sys.path[0] + r"/../")
import time
import torch
import lightning.pytorch as pl
from collections import defaultdict
from utils.utils import motion_temporal_filter
import os
import models
import numpy as np
from tqdm import tqdm

from models.model_utils.data_utils import MotionNormalizerTorch
import shutil
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import models.model_utils.optimizer as optim
import models.model_utils.scheduler as sche

from torch.profiler import profile, ProfilerActivity

from utils.visualize import generate_one_sample
from utils.save import save_pos3d
from utils.metrics_duet_on_training import DuetMetrics

def get_device_from_dict(tensor_dict, default='cpu'):
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device(default)

class ProfilerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.profiler = None

    def on_train_epoch_start(self, trainer, pl_module):
        # 在每个训练周期开始时，启动 profiler
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 选择 CPU 和 GPU 分析
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()

    def on_train_epoch_end(self, trainer, pl_module):
        # 在每个训练周期结束时，停止 profiler 并输出性能数据
        self.profiler.__exit__(None, None, None)
        self.profiler.key_averages()
        

class CopyConfigCallback(pl.Callback):
    def __init__(self, config_paths):
        super().__init__()
        self.config_paths = config_paths if isinstance(config_paths, list) else [config_paths]

    def on_train_start(self, trainer, pl_module):
        # 获取当前的日志目录（事件目录）
        log_dir = trainer.log_dir

        # 检查config文件是否存在
        for config_path in self.config_paths:
            if os.path.exists(config_path):
                shutil.copy(config_path, log_dir)
                print(f"Config file copied to {log_dir}")
            else:
                print(f"Config file {self.config_path} not found!")

class Normalizer_Calculator:
    """
    Utility methods for streaming mean/std computation (Welford's algorithm)
    and persisting normalization statistics.
    """

    @staticmethod
    def init_mean_std():
        """Initialize running mean / M2 accumulator / count."""
        return None, None, 0  # mean, M2 (variance accumulator), sample count

    @staticmethod
    def update_mean_std(data_tensor, mean, M2, count, dim=(0, 1)):
        """Update running mean / variance with a new batch."""
        if data_tensor is None:
            return torch.tensor(0), torch.tensor(1), torch.tensor(1)
        batch_size = data_tensor.size(0)
        count += batch_size
        batch_mean = data_tensor.mean(dim=dim)
        batch_var = data_tensor.var(dim=dim, unbiased=False)

        if mean is None:  # first batch
            mean = batch_mean
            M2 = batch_var * batch_size
        else:
            delta = batch_mean - mean
            mean += delta * batch_size / count
            M2 += batch_var * batch_size + delta ** 2 * (batch_size * (count - batch_size) / count)

        return mean, M2, count

    @staticmethod
    def finalize_std(M2, count):
        """Compute standard deviation from accumulated M2 and count."""
        return torch.sqrt(M2 / count + 1e-8)

    @staticmethod
    def save_mean_std(value_dict, save_dir, save_name='normalizer.pt', overwrite=False):
        """Save normalization statistics to disk, optionally merging with existing file."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, save_name)

        existing_data = {}
        if os.path.isfile(save_path) and not overwrite:
            try:
                device = get_device_from_dict(value_dict)
                existing_data = torch.load(save_path, map_location=device)
                print(f"Loaded existing data for merging.")
                # existing_data 放在后面，从而覆写 value_dict 中相同 key 的值
                merged_data = {**value_dict, **existing_data}
                value_dict = merged_data
            except Exception as e:
                print(f"Warning: Failed to load existing data for merge: {e}")

        torch.save(value_dict, save_path)
        print(f"Saved normalization stats to {save_path}")

# 测试不同的 max_norm 值
def test_clip_grad_norm(model, max_norm_values, norm_type=2):
    import torch.nn as nn
    # 备份原始梯度
    original_grads = {name: param.grad.clone() for name, param in model.parameters()}
    
    results = {}
    for max_norm in max_norm_values:
        # 恢复原始梯度
        for name, param in model.parameters():
            param.grad.copy_(original_grads[name])

        # 模拟梯度裁剪（不修改实际梯度）
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)

        # 保存裁剪后的梯度和总范数
        clipped_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
        results[max_norm] = {
            'total_norm': total_norm.item(),
            'clipped_grads': clipped_grads
        }

    return results

class LitBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters(config)

        # config init
        self.config = config
        self.automatic_optimization = False
        self.accumulate_grad_batches = config.Train.accumulate_grad_batches

        # train settings
        self.use_amp = getattr(self.config.Train, 'use_amp', True)
        self.start_eval_epoch = getattr(self.config.Train, 'start_eval_epoch', 100)
        self.eval_n_epochs = getattr(self.config.Train, 'eval_n_epochs', 10)
        print(f"[###] Use AMP: {self.use_amp}, Start eval epoch: {self.start_eval_epoch}, Eval n epochs: {self.eval_n_epochs}")
        
        ### for pl >= 2.0.0'''
        self.validation_step_outputs = defaultdict(list)
        self.train_accum_loss = 0.
        self.train_num_batches = 0
        
        # dont use Normalizer_Calculator(), it will raise 'TypeError: init_mean_std() takes 0 positional arguments but 1 was given' error
        self.normalizer_calculator = Normalizer_Calculator
        
        use_scaler = getattr(self.config.Train, 'use_scaler', True)
        self.scaler = GradScaler() if use_scaler else None
        self.max_clip_norm = getattr(self.config.Train, 'max_clip_norm', 0.5)
        
        self.use_EMA = getattr(self.config.Train, 'use_EMA', False)
        self.ema_interval = getattr(self.config.Train, 'ema_interval', 1)
        self.num_joints = getattr(self.config.data, 'num_joints', 22)
        
        ## metrics
        self.duet_metric_calc = DuetMetrics(self.num_joints)
        self.last_duet_metrics = defaultdict(list)
        
    def _configure_optim(self):
        try:
            optimizer_config = self.config.Train.optimizer
            optimizer_type = optimizer_config.type
            optim_class = getattr(optim, optimizer_type)
            optim_kwargs = getattr(optimizer_config, optimizer_type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + optimizer_type)
        print("[Optimizer]: ", optimizer_config.type)
        
        try:
            scheduler_config = self.config.Train.scheduler
            scheduler_type = scheduler_config.type
            sched_class = getattr(sche, scheduler_type)
            sched_kwargs = getattr(scheduler_config, scheduler_type)
        except Exception:
            raise NotImplementedError('not implemented scheduler method ' + scheduler_type)
        print("[Scheduler]: ", scheduler_type)
        
        constant_lr = getattr(self.config.Train, 'constant_lr', False)
        optimizer = optim_class(self.model.parameters(), **optim_kwargs) 
        scheduler = sched_class(optimizer=optimizer, **sched_kwargs, verbose=True)
        
        if constant_lr:
            return [optimizer]

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()
    
    def _build(self):
        self._build_models()
        
    def _test_dir_build(self):
        self._dir_setting()

    def _build_models(self):
        model_class = getattr(models, self.config.structure.name)
        model = model_class(self.config.structure, self.num_joints)
        self.model = model

    def _dir_setting(self):
        from utils.config_utils import build_experiment_dirs

        self.expname = self.config.expname
        checkpoint_path = self.config.data.test.checkpoint
        self.expdir, self.sampledir = build_experiment_dirs(
            self.expname, checkpoint_path
        )
        
    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.config.Train.last_iter if self.config.Train.last_iter else 0
        self.logs = defaultdict(float)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss, loss_logs, metrics = self.forward(batch)

        self.log('TRAIN_LOSS', loss, prog_bar=True, sync_dist=True)
        self._backward_and_step(loss, opt, batch_idx)
        self._update_ema(batch_idx)

        self.train_accum_loss += loss.item()
        self.train_num_batches += 1  # 更新批次数量
        
        return {"TRAIN_LOSS": loss,
                "loss_logs": loss_logs,
                "metrics": metrics
                }

    def _backward_and_step(self, loss, opt, batch_idx: int) -> None:
        """
        Handle backward pass, gradient accumulation, optional AMP/GradScaler,
        and optimizer step with optional gradient clipping.
        """
        if self.scaler is not None:
            self.scaler.scale(loss / self.accumulate_grad_batches).backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.max_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_norm)
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad()
        else:
            self.manual_backward(loss / self.accumulate_grad_batches)

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.max_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_norm)
                opt.step()
                opt.zero_grad()

    def _update_ema(self, batch_idx: int) -> None:
        """Update EMA weights if enabled."""
        if not self.use_EMA:
            return
        if (batch_idx + 1) % self.ema_interval != 0:
            return

        self.model.diffusion.ema.update_model_average(
            self.model.diffusion.master_model, self.model
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            self.logs[k] += v.item()

        for k, v in outputs['metrics'].items():
            self.logs[k] += v.item()

        self.it += 1
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(name='lr', value=lr, prog_bar=False, logger=True, sync_dist=True)
        if self.it % self.config.Train.log_steps == 0:
            mean_loss = dict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.config.Train.log_steps
                log_tag = tag if self.config.expname == "HFSQ" else f"Train/{tag}"
                self.log(name=log_tag, value=mean_loss[tag], prog_bar=False, logger=True, sync_dist=True)
            self.logs = defaultdict(float)
        
    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
            
        if self.current_epoch !=0 and self.current_epoch % 200 == 0:
            checkpoint_path = f'{self.trainer.checkpoint_callback.dirpath}/manual_epoch_{self.current_epoch}_checkpoint.ckpt'
            print(f"Saving checkpoint for epoch {self.current_epoch} at {checkpoint_path}")
            self.trainer.save_checkpoint(checkpoint_path)  # 保存 checkpoint
            
    def validation_step(self, batch, batch_idx):
        loss, loss_logs, metrics = self(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        output = {"val_loss": loss}
        
        # for key, value pair in loss_logs, append self.validation_step_outputs
        for key, value in loss_logs.items():
            self.validation_step_outputs[key].append(value)

        for key, value in metrics.items():
            self.validation_step_outputs[key].append(value)

        return output

    def on_validation_epoch_end(self):
        if self.train_num_batches != 0:
            mean_epoch_loss = self.train_accum_loss / self.train_num_batches
            self.log('TEM_LOSS', mean_epoch_loss, prog_bar=False, sync_dist=True)    # TEM_LOSS: Train Epoch Mean Loss
            
            # 清空累积损失和批次数量，准备下一个 epoch
            self.train_accum_loss = 0.
            self.train_num_batches = 0
        
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        for loss_name, loss_values in self.validation_step_outputs.items():
            if len(loss_values) != 0:
                avg_loss = torch.stack(loss_values).mean()
                if self.config.expname == "HFSQ":
                    self.log(f'Validation_{loss_name}', avg_loss, sync_dist=True)
                else:
                    self.log(f'Validation/{loss_name}', avg_loss, sync_dist=True)
            
            if loss_name == 'total':
                self.log(f'VAL_LOSS', avg_loss, sync_dist=True)
            
        # clear validation loss dict
        self.validation_step_outputs = defaultdict(list)
        
    def eval_during_training(self, **kwargs):
        jointsl_list, jointsf_list, fname_list = self.calc_duet_joints(**kwargs)
        duet_metrics = self.duet_metric_calc.eval_duet_metrics(jointsl_list, jointsf_list, fname_list)
        
        print(f'[Real Eval Metrics]: FID: {duet_metrics["fid_k"]}, MPJPE: {duet_metrics["mpjpe"]}')
        self.log(f'fid_k', duet_metrics['fid_k'], prog_bar=True, logger=True, sync_dist=False)
        self.log(f'mpjpe', duet_metrics['mpjpe'], prog_bar=True, logger=True, sync_dist=False)
        self.last_duet_metrics = duet_metrics

    def calc_duet_joints(self, **kwargs):
        # implement in subclass
        pass
            
    def _single_synthesis(self, pose_seq):
        pose_seq = pose_seq.cpu().data.numpy()

        return pose_seq
    
    def _duet_synthesis(self, pose_sample, pose_seql, translf, rootl=None):
        b, n, _ = pose_sample.shape
        if rootl is not None:
            pose_seql = pose_seql.view(b, n, -1, 3) + rootl.unsqueeze(-2)
            pose_seql[:, :, 0, :] = rootl   # for both pos_zero_root and pos_pelvis_root
            pose_seql = pose_seql.view(b, n, -1)

        pose_sample[:, :, :3] = 0
        root_offset = pose_seql[:, :, :3] + translf
        pose_sample = pose_sample.view(b, n, -1, 3) + root_offset.unsqueeze(-2)
        pose_sample = pose_sample.view(b, n, -1).cpu().data.numpy()
        pose_seql = pose_seql.view(b, n, -1).cpu().data.numpy()

        return pose_sample, pose_seql

    def vis_gt(self):
        gt_dir = 'data_lazy/motion/GT_vis'
        os.makedirs(gt_dir, exist_ok=True)

        with torch.no_grad():
            print("Mix GT Vis...")

            followers = []
            leaders = []
            dance_names = []
            for i_eval, batch in enumerate(tqdm(self.test_dl, desc='Generating Dance Poses')):
                pose_seqf, pose_seql = batch['pos3df'].to(self.device), batch['pos3dl'].to(self.device)
                fname = batch['fname'][0]
                
                translf = pose_seqf[:, :, :3] - pose_seql[:, :, :3]
                pose_seqf = (pose_seqf.view(pose_seqf.size(0), pose_seqf.size(1), -1, 3) - pose_seqf[:, :, :3].unsqueeze(-2)).view(pose_seqf.size(0), pose_seqf.size(1), -1)

                pose_seqf, pose_seql = self._duet_synthesis(pose_seqf, pose_seql, translf)
                dance_names.append(fname)
                followers.append(pose_seqf)
                leaders.append(pose_seql)
                
                pose_sample = pose_seqf.squeeze(0)
                pose_seql = pose_seql.squeeze(0)
                motion_multiple = [pose_seql, pose_sample]
                
                generate_one_sample(motion_multiple, fname, gt_dir, nb_joints=self.num_joints)

class LitReactDance_hfsq(LitBaseModel):
    def __init__(self, config, test_loader=None):
        super().__init__(config)
        
        self.start_eval_epoch = getattr(self.config.Train, 'start_eval_epoch', 0)
        self.eval_n_epochs = getattr(self.config.Train, 'eval_n_epochs', 1)
        self.start_vis_epoch = getattr(self.config.Train, 'start_vis_epoch', 1)
        self.vis_n_epochs = getattr(self.config.Train, 'vis_n_epochs', 20)
        self.test_dl = test_loader
        self._build()
        
    def _build(self):
        self._set_params()  # should before super()._build()
        super()._build()

    def _set_params(self):
        # body part yaml setting
        unified_params = getattr(self.config.structure, 'unified_params', None)
        if unified_params is not None:
            block_keys = ['up_half', 'down_half', 'tr_half']
            
            for block_key in block_keys:
                if block_key in self.config.structure:
                    for u_k, u_v in unified_params.items():
                        if u_k not in self.config.structure[block_key].keys():
                            self.config.structure[block_key][u_k] = u_v
                        
            print(f'MQ Set Params Done!')
        else:
            print(f'Skip MQ Params Set!')
     
    # @torch.compile(mode='max-autotune', fullgraph=True) # useful
    def forward(self, batch):
        with autocast(enabled=self.use_amp):
            pos_output, losses, metrics = self.model(batch)

        return losses['total'], losses, metrics
    
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        # main eval
        if self.current_epoch >= self.start_eval_epoch and self.current_epoch % self.eval_n_epochs == 0:
            print(f'Eval after validation at epoch {self.current_epoch}')
            self.eval_during_training()

        # vis during training
        if self.current_epoch >= self.start_vis_epoch and self.current_epoch % self.vis_n_epochs == 0:
            print(f'Vis after training at epoch {self.current_epoch}')
            self.synthesis_and_vis(self.current_epoch)
            
    def calc_duet_joints(self, **kwargs):
        device = self.device
        with torch.no_grad(), autocast(enabled=self.use_amp):
            model = self.model.to(device)
            
            print("Stage One: calc_duet_joints...", flush=True)
            followers = []
            leaders = []
            fname_list = []
            for i_eval, batch in enumerate(tqdm(self.test_dl, desc='calc_duet_joints')):
                follower_mode = getattr(self.config.structure.unified_params.Bottleneck.HFSQ_BottleneckBlock.RFSQ, "prior_mode", "all_quantizeds")
                batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                latentf = model.encode(batch, encode_mode=follower_mode)
                pos_output = model.decode(latentf, decode_mode=follower_mode)

                pos3df_pred = pos_output['pos3df']
                translf_pred = pos_output['translf']

                pose_sample, pose_seql = self._duet_synthesis(pos3df_pred, batch['pos3dl'], translf_pred)
                followers.append(pose_sample)
                leaders.append(pose_seql)
                fname_list.append(batch['fname'][0])
            return leaders, followers, fname_list
        
    def duet_synthesis(self, no_video=False, device='cuda'):
        self._test_dir_build()
        with torch.no_grad():
            model = self.model.to(device).eval()
            
            print("Stage One: duet_synthesis...", flush=True)
            followers = []
            leaders = []
            dance_names = []
            motion_multiples = []
            for i_eval, batch in enumerate(tqdm(self.test_dl, desc='MQ Generating Dance Poses')):
                batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                fname = batch['fname'][0]
                dance_names.append(fname)

                follower_mode = "all_quantizeds"
                latentf = model.encode(batch, encode_mode=follower_mode)
                pos_output = model.decode(latentf, decode_mode=follower_mode)
                
                save_dir = os.path.join(self.sampledir, 'duet')
                pose_sample, pose_seql = self._duet_synthesis(pos_output['pos3df'], batch['pos3dl'], pos_output['translf'])
                followers.append(pose_sample)
                leaders.append(pose_seql)
                motion_multiples.append([pose_seql, pose_sample])
                
                save_pos3d(pose_sample, pose_seql, save_dir, fname)
                
        if not no_video:
            video_dir = os.path.join(save_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            for motion_multiple, fname in zip(motion_multiples, dance_names):
                generate_one_sample(motion_multiple, fname, video_dir, nb_joints=self.num_joints)
                
    def synthesis_and_vis(self, epoch, result_dir=None):
        if result_dir is None:
            output_dir = os.path.join(self.trainer.log_dir, "synthesis_log")
        else:
            output_dir = os.path.join(result_dir, "test_synthesis_log")
        npy_dir = os.path.join(output_dir, "npy")
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        def synthesis_and_vis_one_sample(epoch, dataset, sample_num=2):
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    print("HFSQ Synthesis...", flush=True)
                    dance_names = []
                    for i_eval, batch in enumerate(tqdm(dataset, desc='Generating Dance Poses')):
                        batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        fname = batch['fname'][0]
                        dance_names.append(fname)

                        follower_mode = "all_quantizeds"
                        latentf = self.model.encode(batch, encode_mode=follower_mode)
                        pos_output = self.model.decode(latentf, decode_mode=follower_mode)
                        pose_sample, pose_seql = self._duet_synthesis(pos_output['pos3df'], batch['pos3dl'], pos_output['translf'])

                        pose_sample = pose_sample.squeeze(0)
                        pose_seql = pose_seql.squeeze(0)
                        motion_multiple = [pose_seql, pose_sample]
                        generate_one_sample(motion_multiple, f"{fname}_epoch{epoch}", video_dir, nb_joints=self.num_joints)
                        
                        if len(dance_names) >= sample_num:
                            break

        synthesis_and_vis_one_sample(epoch, self.test_dl, sample_num=1)                    

    def save_mean_std(self, train_loader, offline=False, device='cuda', modes=["quantizeds", "all_quantizeds"]):
        normalizer_calculator = self.normalizer_calculator
        Normalizer_dict = {}
        
        with torch.no_grad():
            model = self.model.to(device).eval()
            for mode in modes:
                if offline:
                    # Offline 模式: 收集所有数据然后一次性计算
                    up_list = []
                    down_list = []
                    tr_list = []
                    for batch in tqdm(train_loader, desc=f'collect {mode} data'):
                        batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        encoded_x = model.encode(batch, encode_mode=mode)
                        xup, xdown, xtr = encoded_x
                        
                        up_list.append(xup.cpu())
                        down_list.append(xdown.cpu())
                        tr_list.append(xtr.cpu())
                    # 拼接数据并计算统计量
                    all_up = torch.cat(up_list, dim=1)
                    all_down = torch.cat(down_list, dim=1)
                    all_tr = torch.cat(tr_list, dim=1)
                    up_mean = all_up.mean(dim=(0,1))
                    up_std = all_up.std(dim=(0,1))
                    down_mean = all_down.mean(dim=(0,1))
                    down_std = all_down.std(dim=(0,1))
                    tr_mean = all_tr.mean(dim=(0,1))
                    tr_std = all_tr.std(dim=(0,1))
                else:
                    # Online 模式: 使用 Welford's algorithm 增量计算
                    up_mean, up_M2, up_count = normalizer_calculator.init_mean_std()
                    down_mean, down_M2, down_count = normalizer_calculator.init_mean_std()
                    tr_mean, tr_M2, tr_count = normalizer_calculator.init_mean_std()
                    
                    for batch in tqdm(train_loader, desc=f'calc {mode} normalizer'):
                        batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        z = model.encode(batch, encode_mode=mode)
                        zup, zdown, ztr = z
                        up_mean, up_M2, up_count = normalizer_calculator.update_mean_std(zup, up_mean, up_M2, up_count)
                        down_mean, down_M2, down_count = normalizer_calculator.update_mean_std(zdown, down_mean, down_M2, down_count)
                        tr_mean, tr_M2, tr_count = normalizer_calculator.update_mean_std(ztr, tr_mean, tr_M2, tr_count)
                        
                    up_std = normalizer_calculator.finalize_std(up_M2, up_count)
                    down_std = normalizer_calculator.finalize_std(down_M2, down_count)
                    tr_std = normalizer_calculator.finalize_std(tr_M2, tr_count)
                
                mean_val = torch.cat([up_mean, down_mean, tr_mean], dim=-1)
                std_val = torch.cat([up_std, down_std, tr_std], dim=-1)

                # 存储结果
                normalizer_dict = {
                    f'{mode}-mean': mean_val,
                    f'{mode}-std': std_val,
                }
                Normalizer_dict.update(normalizer_dict)
            
            # 保存结果
            save_dir = self.config.data.test.checkpoint.replace('checkpoints', 'normalizers')[:-5]
            save_name = 'normalizer.pt'
            normalizer_calculator.save_mean_std(Normalizer_dict, save_dir, save_name)
            print(f'Save Mean and Std of {mode} of MQ Done')
            
class LitReactDance_reactdance(LitBaseModel):
    def __init__(self, config, val_dl, test_dl):
        super().__init__(config)

        self.start_eval_epoch = getattr(self.config.Train, 'start_eval_epoch', 0)
        self.eval_n_epochs = getattr(self.config.Train, 'eval_n_epochs', 1)
        self.start_vis_epoch = getattr(self.config.Train, 'start_vis_epoch', 10)
        self.vis_n_epochs = getattr(self.config.Train, 'vis_n_epochs', 10)

        self.val_dl = val_dl
        self.test_dl = test_dl
        self._build()

    def _build(self):
        self._build_models()

    def _build_models(self):
        from utils.checkpoint_loading import cfg_reader

        config = self.config
        config_name = config.structure_generate.generator_name
        gen_class = getattr(models, config_name)
        self.generator_config = config.structure_generate.get(config_name, dict())
        assert len(self.generator_config) > 0, f"config {config_name} not found"
        calculator = getattr(self.generator_config, 'normalizer_calculator', "Z_SCORE")
        
        def __load_weight():
            lit_hfsq_cfg = cfg_reader(config.HFSQ_config)
            lit_hfsq_checkpoint_path = lit_hfsq_cfg.data.test.checkpoint
            gen_normalizer = self._set_normalizer(
                calculator = calculator,
                hfsq_normalizer_dir=lit_hfsq_checkpoint_path.replace('checkpoints', 'normalizers')[:-5],            # remove '.ckpt'
            )
            
            # hfsq
            try:
                lit_hfsq_checkpoint = torch.load(lit_hfsq_checkpoint_path, map_location='cuda')
                lit_hfsq = LitReactDance_hfsq(lit_hfsq_cfg) 
                lit_hfsq.load_state_dict(lit_hfsq_checkpoint['state_dict'])
                hfsq_model = lit_hfsq.model
                print("[GENERATOR TRAINING] Motion loaded from pl checkpoint")
            except Exception as e:
                # remember to check device setting yaml
                raise NotImplementedError(f"[GENERATOR TRAINING] HFSQ Model Loading Error: {e}")
            
            return hfsq_model, gen_normalizer
        
        # load weight before build generator model
        hfsq, gen_normalizer = __load_weight()
        self.hfsq = hfsq.cuda().eval()

        decoders = dict(hfsq = self.hfsq)
        
        generator = gen_class(self.generator_config, normalizer=gen_normalizer, decoders=decoders, use_EMA=self.use_EMA)
        self.model = generator.cuda().train()
        self.gen_normalizer = gen_normalizer
        
        self.prior_mode = getattr(self.generator_config, "prior_mode", "all_quantizeds")
        self.use_pred_as_target = getattr(self.config.Train, "use_pred_as_target", True)

    def _set_normalizer(self, calculator, hfsq_normalizer_dir):
        normalizer = MotionNormalizerTorch(
            calculator=calculator,
            hfsq_normalizer_file=os.path.join(hfsq_normalizer_dir, "normalizer.pt"),
        )
        
        return normalizer

    # @torch.compile(mode='max-autotune', fullgraph=True) # useful
    def forward(self, batch):
        with autocast(enabled=self.use_amp):
            pose_seql, pose_seqf, music_cond = batch["pos3dl"], batch["pos3df"], batch["music"]
            pose_seql, _ = self.hfsq.rootl_full_forward(pose_seql)
            zout = self.hfsq.encode(batch, encode_mode=self.prior_mode)
            cond_tuple = (music_cond, pose_seql)
                    
            if not self.use_pred_as_target:
                motion_target = dict(
                    pos3df=pose_seqf,
                    translf=pose_seqf[:, :, :3] - pose_seql[:, :, :3],
                )
            else:
                posf_pred = self.hfsq.decode(zout, decode_mode=self.prior_mode)
                pos3df_pred, translf_pred = posf_pred['pos3df'], posf_pred['translf']
                motion_target = dict(
                    pos3df=pos3df_pred,
                    pos3dl=batch["pos3dl"],
                    translf=translf_pred,
                )

            output, losses, metrics = self.model(
                x_start = zout,
                cond_tuple = cond_tuple,
                motion_target = motion_target,
            )
            
        return losses['total'], losses, metrics
    
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.last_duet_metrics.get('fid_k', None) is None or (self.current_epoch >= self.start_eval_epoch and self.current_epoch % self.eval_n_epochs == 0):
            self.eval_during_training()
            
        # vis during training
        if self.current_epoch >= self.start_vis_epoch and self.current_epoch % self.vis_n_epochs == 0:
            print(f'Vis after training at epoch {self.current_epoch}')
            self.synthesis_and_vis(self.current_epoch)
    
    def calc_duet_joints(self):
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                hfsq = self.hfsq
                generator = self.model
                
                jointsl_list = []
                jointsf_list = []
                fname_list = []
                for i_eval, batch in enumerate(tqdm(self.test_dl, desc=f'[*] ReactDance calc_duet_joints')):
                    batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    music_seq, pose_seql, fname = batch['music'], batch['pos3dl'], batch['fname'][0]
                    
                    pose_seql, _ = self.hfsq.rootl_full_forward(pose_seql)
                    cond_tuple = (music_seq, pose_seql)
                 
                    pose_pred_emb = generator.sample(
                        use_EMA = self.use_EMA,
                        cond_tuple=cond_tuple, 
                        sample_after_train=False,
                    )
                    pose_sample = hfsq.decode(pose_pred_emb, decode_mode=self.prior_mode)
                    
                    pos3df = pose_sample['pos3df']
                    translf = pose_sample['translf']

                    pose_sample, pose_seql = self._duet_synthesis(pos3df, batch['pos3dl'], translf)
                    assert not np.isnan(pose_sample).any(), "pose_sample is nan"
                    jointsl_list.append(pose_seql)
                    jointsf_list.append(pose_sample)
                    fname_list.append(fname)
            return jointsl_list, jointsf_list, fname_list
                
    def synthesis(
        self, 
        test_loader, 
        sample_after_train=True, 
        no_video=True, 
        no_dir_build=False, 
        pm_guidance_weight=5,
        **kwargs
        ):
        if not no_dir_build:
            self._test_dir_build()
        
        output_dir = self.sampledir
        time_write_file = os.path.join(output_dir, f"time.txt")
        
        # 初始化计时器
        total_forward_time = 0.0
        
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                hfsq = self.hfsq.eval()
                generator = self.model.eval()
                
                print("GPT Synthesis...", flush=True)
                followers = []
                leaders = []
                dance_names = []
                motion_multiples = []
                for i_eval, batch in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                    batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    music_seq, pose_seql, pose_seqf = batch['music'], batch['pos3dl'], batch['pos3df']
                    
                    # ⏱️ 开始计时
                    start_time = time.perf_counter()
                    fname = batch['fname'][0]
                    dance_names.append(fname)
                    
                    pose_seql, _ = self.hfsq.rootl_full_forward(pose_seql)
                    
                    cond_tuple = (music_seq, pose_seql)
                    pose_pred_emb = generator.sample_cfg(
                        cond_tuple=cond_tuple,
                        sample_after_train=sample_after_train,
                        pm_guidance_weight=pm_guidance_weight,
                    )
                        
                    pose_sample = hfsq.decode(pose_pred_emb, decode_mode=self.prior_mode)
                    pose_sample, pose_seql = self._duet_synthesis(pose_sample['pos3df'], batch['pos3dl'], pose_sample['translf'])
                    pose_sample = motion_temporal_filter(pose_sample)
                        
                    # ⏱️ 结束计时
                    end_time = time.perf_counter()
                    # 累加耗时
                    total_forward_time += (end_time - start_time)
                    motion_multiples.append([pose_sample, pose_seql])
                    
                    followers.append(pose_sample)
                    leaders.append(pose_seql)
                    
                    save_pos3d(pose_sample, pose_seql, output_dir, fname)
                
        with open(time_write_file, "w") as f:
            f.write(f"Total time: {total_forward_time:.4f} seconds, average forward time: {total_forward_time / len(dance_names)} seconds")
        print(f"[INFO] Total time: {total_forward_time:.4f} seconds, average forward time: {total_forward_time / len(dance_names)} seconds")
            
        if not no_video:
            video_dir = os.path.join(output_dir, f'videos')
            os.makedirs(video_dir, exist_ok=True)
            for motion_multiple, fname in zip(motion_multiples, dance_names):
                generate_one_sample(motion_multiple, fname, video_dir, nb_joints=self.num_joints)
        
        return followers, leaders, dance_names, os.path.dirname(output_dir)
    
    def synthesis_and_vis(self, epoch, result_dir=None):
        hfsq = self.hfsq
        generator = self.model
        if result_dir is None:
            output_dir = os.path.join(self.trainer.log_dir, "synthesis_log")
        else:
            output_dir = os.path.join(result_dir, "test_synthesis_log")
        npy_dir = os.path.join(output_dir, "npy")
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        def synthesis_and_vis_one_sample(epoch, dataset, dataset_name, sample_num=2):
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    print("ReactDance Synthesis...", flush=True)
                    dance_names = []
                    followers = []
                    leaders = []
                    
                    for i_eval, batch in enumerate(tqdm(dataset, desc='Generating Dance Poses')):
                        batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        music_seq, pose_seql = batch['music'], batch['pos3dl']

                        fname = batch['fname'][0]
                        dance_names.append(f'{fname}_{dataset_name}_epoch{epoch}_duovis')
                        
                        pose_seql, _ = self.hfsq.rootl_full_forward(pose_seql)
                        cond_tuple = (music_seq, pose_seql)
                        
                        pose_pred_emb = generator.sample(
                            cond_tuple=cond_tuple, 
                            sample_after_train=False,
                        )

                        pose_sample = hfsq.decode(pose_pred_emb, decode_mode=self.prior_mode)
                        pose_sample, pose_seql = self._duet_synthesis(pose_sample['pos3df'], batch['pos3dl'], pose_sample['translf'])
                        pose_sample = pose_sample[0]
                        pose_seql = pose_seql[0]
                        motion_multiple = [pose_seql, pose_sample]
                        generate_one_sample(motion_multiple, f"{fname}_{dataset_name}_epoch{epoch}_intvis", video_dir, nb_joints=self.num_joints)
                        followers.append(pose_sample)
                        leaders.append(pose_seql)
                        
                        if len(dance_names) >= sample_num:
                            break
                        
                    return followers, leaders, dance_names
                    
        followers, leaders, dance_names =synthesis_and_vis_one_sample(epoch, self.val_dl, 'val', sample_num=2)
        followers, leaders, dance_names = synthesis_and_vis_one_sample(epoch, self.test_dl, 'test', sample_num=1)
        
        return followers, leaders, dance_names