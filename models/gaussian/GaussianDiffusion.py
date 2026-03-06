import copy
import sys
sys.path.append("..")
import pickle
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from tqdm import tqdm
from models.nets.losses import LatentLoss, GeometricLoss

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        normalizer,
        decoders,
        n_timestep=1000,
        predict_epsilon=False,
        geo_loss_checklist = None,
        config = None,
        use_EMA = False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        # self.ema = EMA(0.9999)
        # if use_EMA:
        #     self.master_model = copy.deepcopy(self.model)
        schedule = getattr(config, "beta_scheduler", "cosine")
        
        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        use_p2 = getattr(config, "use_p2", False)
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.05 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        loss_fn = getattr(config, "loss_fn", "l1")

        self.normalizer = normalizer
        self.decoders = decoders
        self.prior_mode = getattr(config, "prior_mode", None)
        
        geo_loss_checklist = geo_loss_checklist if geo_loss_checklist is not None else {}
        self.geometry_loss_manager = GeometricLoss(loss_fn, n_joints=22, loss_checklist=geo_loss_checklist)
        self.latent_loss_manager = LatentLoss()
        
        # shape 配置项
        self.latent_loss_weight = torch.tensor(getattr(config, 'latent_loss_weight', [2., 1.]), device='cuda')
        self.tr_latent_loss_weight = torch.tensor(getattr(config, 'tr_latent_loss_weight', [2., 1.]), device='cuda')
        self.pos_allq_shape = getattr(config, 'pos_allq_shape', [8, 3, 128])
        self.pos_allq_shape[-1] //= 3
        self.clip_scaler = getattr(config, 'clip_scaler', 1.0)
            
    # ------------------------------------------ sampling ------------------------------------------#
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond_tuple, t, use_EMA, clip_denoised = False, is_last_step=False, model_kwargs=None):
        model = self.master_model if use_EMA else self.model
        
        sample_func = getattr(model, "guided_forward") if model_kwargs.get("pm_guidance_weight", 1) > 1 else getattr(model, "forward")
        model_output = sample_func(x, cond_tuple, t, is_last_step=is_last_step, **model_kwargs)
        maybe_clip = partial(torch.clamp, min = -self.clip_scaler, max = self.clip_scaler) if clip_denoised else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond_tuple, t, use_EMA=False, model_kwargs=None):
        # # guidance clipping
        # if t[0] > 1.0 * self.n_timestep:
        #     weight = min(self.guidance_weight, 0)
        # elif t[0] < 0.1 * self.n_timestep:
        #     weight = min(self.guidance_weight, 1)
        # else:
        #     weight = self.guidance_weight

        model = self.master_model if use_EMA else self.model
        x_recon = self.predict_start_from_noise(
            # x, t=t, noise=model.guided_forward(x, cond_tuple, t, **model_kwargs)
            x, t=t, noise=model.forward(x, cond_tuple, t, **model_kwargs)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon
    
    @torch.no_grad()
    def p_sample(self, x, cond_tuple, t, use_EMA=False, model_kwargs=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond_tuple=cond_tuple, t=t, use_EMA=use_EMA, model_kwargs=model_kwargs
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond_tuple,
        noise=None,
        return_diffusion=False,
        start_point=None,
        use_EMA=False,
        progress=False,
        model_kwargs=None,
        **kwargs,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond_tuple = self.cond_emb_normalize(cond_tuple)

        if return_diffusion:
            diffusion = [x]

        if progress:
            time_pairs = tqdm(reversed(range(0, start_point)), desc = 'sampling loop time step')
        else:
            time_pairs = reversed(range(0, start_point))

        for i in time_pairs:
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond_tuple, timesteps, use_EMA=use_EMA, model_kwargs=model_kwargs)

            if return_diffusion:
                diffusion.append(x)

        pred_pose_tuple = self.motion_emb_denormalize(x, input_mode="tensor")
        return pred_pose_tuple
    
    @torch.no_grad()
    def ddim_sample_loop(self, shape, cond_tuple, use_EMA, progress=False, sampling_timesteps=50, eta=0.0, clip_denoised=False, model_kwargs=None):
        batch, device, total_timesteps = shape[0], self.betas.device, self.n_timestep
        
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))[1:] # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        cond_tuple = self.cond_emb_normalize(cond_tuple)
        x = torch.randn(shape, device = device)
        
        x_start = None

        if progress:
            # Lazy import so that we don't depend on tqdm.
            time_pairs = tqdm(time_pairs, desc = 'sampling loop time step')
            
        for i, (time, time_next) in enumerate(time_pairs):
            is_last_step = i == len(time_pairs) - 1
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x, 
                cond_tuple, 
                time_cond, 
                use_EMA = use_EMA,
                clip_denoised = clip_denoised,
                is_last_step = is_last_step,
                model_kwargs = model_kwargs,
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        pred_pose_tuple = self.motion_emb_denormalize(x, input_mode="tensor")
        return pred_pose_tuple

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def cond_emb_normalize(self, cond_tuple):
        cond_music, cond_pos = cond_tuple
        cond_music = self.normalizer.forward(cond_music, "music")
        cond_pos = self.normalizer.forward(cond_pos, "pos3dl")
        
        return (cond_music, cond_pos)
        
    def motion_emb_normalize(self, x_start):
        if isinstance(x_start, tuple):
            x_start_up, x_start_down, x_start_tr = x_start
            if self.prior_mode in ['all_quantizeds']:
                b, n = x_start_up.shape[:2]
                x_start_up = x_start_up.view(b, n, -1)
                x_start_down = x_start_down.view(b, n, -1)
                x_start_tr = x_start_tr.view(b, n, -1)
            x_start = torch.cat([x_start_up, x_start_down, x_start_tr], dim=-1)
            
        mq_suffix = f"{self.prior_mode}"
        x_emb = self.normalizer.forward(x_start, mq_suffix)

        return x_emb

    def value_check(self, tensor, name):
        assert not torch.isnan(tensor).any(), f"{name} contain NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contain infinite values"

    def motion_emb_denormalize(self, pred, input_mode="tuple"):
        if input_mode == "tuple":
            pred_emb_up, pred_emb_down, pred_emb_tr = pred
            pred_emb = torch.cat([pred_emb_up, pred_emb_down, pred_emb_tr], dim=-1)
        elif input_mode == "tensor":
            pred_emb = pred
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")
            
        mq_suffix = f"{self.prior_mode}"
        pred_emb = self.normalizer.backward(pred_emb, mq_suffix)
        pred_emb_up, pred_emb_down, pred_emb_tr = torch.split(pred_emb, pred_emb.shape[-1]//3, dim=-1)
                
        return (pred_emb_up, pred_emb_down, pred_emb_tr)
    
    # region calc_separate_loss
    def calc_separate_loss(self, model_output, target, keepdim=False):
        pred_emb = model_output
        target_emb = target
        
        b, n = target_emb.shape[:2]
        pred_emb_up, pred_emb_down, pred_emb_tr = torch.split(pred_emb, pred_emb.shape[-1]//3, dim=-1)
        target_emb_up, target_emb_down, target_emb_tr = torch.split(target_emb, target_emb.shape[-1]//3, dim=-1)
        
        if self.prior_mode in ["all_quantizeds"]:
            mse_up = torch.mean(torch.mean(((target_emb_up - pred_emb_up) ** 2).view(b, n, *self.pos_allq_shape), dim=-1) * self.latent_loss_weight)
            mse_down = torch.mean(torch.mean(((target_emb_down - pred_emb_down) ** 2).view(b, n, *self.pos_allq_shape), dim=-1) * self.latent_loss_weight)
            mse_tr = torch.mean(torch.mean(((target_emb_tr - pred_emb_tr) ** 2).view(b, n, *self.pos_allq_shape), dim=-1) * self.tr_latent_loss_weight)
        else:
            mse_up = torch.mean(target_emb_up - pred_emb_up) ** 2
            mse_down = torch.mean(target_emb_down - pred_emb_down) ** 2
            mse_tr = torch.mean(target_emb_tr - pred_emb_tr) ** 2
            
        mse = mse_up + mse_down + mse_tr
            
        return mse, pred_emb
    # endregion
    
    def p_losses(self, x_start, cond, t, model_kwargs=None, mode=None):
        terms = {}
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, cond_tuple=cond, timesteps=t, **model_kwargs)
        assert noise.shape == x_recon.shape
        self.value_check(x_recon, "x_recon")

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss, pred = self.calc_separate_loss(model_out, target, keepdim=True)
        
        terms["target"] = target
        terms["loss"] = loss
        terms["pred"] = pred

        return terms

    def forward(self, x_start, cond_tuple, motion_target, train_mode='train', t_override=None, **kwargs):
        x_start = self.motion_emb_normalize(x_start)
        cond_tuple = self.cond_emb_normalize(cond_tuple)
        
        losses = dict(total=0)
        metrics = dict()
        batch_size = len(x_start)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x_start.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x_start.device).long()
            
        items = self.p_losses(x_start, cond_tuple, t, **kwargs)
        
        # normalize backward
        pred_emb_pose_tuple = self.motion_emb_denormalize(items["pred"], input_mode="tensor")
        hfsq_model = self.decoders['hfsq']
        posf_pred = hfsq_model.decode(pred_emb_pose_tuple, decode_mode=self.prior_mode)
        pos3df_pred, translf_pred = posf_pred['pos3df'], posf_pred['translf']

        motion_pred = dict(
            pos3df=pos3df_pred,
            translf=translf_pred,
        )
        
        if train_mode != "sample":
            self.latent_loss_manager.forward(items["loss"])
            losses['total'] = self.latent_loss_manager.losses['latent_loss_total']
            
            self.geometry_loss_manager.forward(motion_pred, motion_target)
            # self.inter_loss_manager.forward(motion_pred, motion_target, timestep_mask=pose_timestep_mask)
            
            losses.update(self.latent_loss_manager.losses)
            losses.update(self.geometry_loss_manager.losses)
            # losses.update(self.inter_loss_manager.losses)
            metrics.update(self.geometry_loss_manager.metrics)
            # metrics.update(self.inter_loss_manager.metrics)
            
            losses['total'] += self.geometry_loss_manager.losses['geometry_loss_total']
            # losses['total'] += self.geometry_loss_manager.losses['geometry_loss_total'] + self.inter_loss_manager.losses['inter_loss_total']
        else:
            losses = {}
            metrics = {}
        
        return pred_emb_pose_tuple, losses, metrics
    
    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x
