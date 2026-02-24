"""
SEAL Adapter for DIT-BDH

"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

try:
    from src.net.diffusion import GaussianDiffusion
    from src.net.ssim import SSIM
except ModuleNotFoundError:
    try:
        from diffusion import GaussianDiffusion
        from ssim import SSIM
    except ModuleNotFoundError:
        GaussianDiffusion = None
        SSIM = None


@dataclass
class SEALConfig:
    """Configuration for SEAL adaptation."""
    ttt_lr: float = 1e-4
    ttt_steps: int = 1
    confidence_threshold: float = 0.5
    consistency_threshold: float = 0.7
    use_momentum: bool = False
    momentum: float = 0.0
    weight_decay: float = 0.0
    max_diffusion_steps: int = 50
    diffusion_schedule: str = 'cosine'
    adapt_layers: str = 'all'
    temperature: float = 1.0
    ema_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    use_cosine_schedule: bool = False
    reset_optimizer_per_sample: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class ConsistencyChecker:
    """Validates pseudo-labels via self-consistency."""
    
    def __init__(self, ssim_threshold: float = 0.7, variance_threshold: float = 0.1):
        self.ssim_threshold = ssim_threshold
        self.variance_threshold = variance_threshold
        self.ssim_module = SSIM(data_range=2.0, size_average=True, win_size=11, channel=3) if SSIM else None
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        if self.ssim_module:
            return self.ssim_module((img1+1)/2, (img2+1)/2).item()
        return 1.0 / (1.0 + F.mse_loss(img1, img2).item())
    
    def check_consistency(self, input_image: torch.Tensor, prediction: torch.Tensor,
                         additional_predictions: Optional[List[torch.Tensor]] = None) -> Tuple[bool, Dict]:
        metrics = {'ssim': self.compute_ssim(input_image, prediction)}
        if additional_predictions:
            stacked = torch.stack([prediction] + additional_predictions, dim=0)
            metrics['variance'] = torch.var(stacked, dim=0).mean().item()
        else:
            metrics['variance'] = 0.0
        is_valid = self.ssim_threshold < metrics['ssim'] < 0.99 and metrics['variance'] < self.variance_threshold
        metrics['is_valid'] = is_valid
        return is_valid, metrics


class SEALAdapter(nn.Module):
    """SEAL Adapter for Test-Time Training - CORRECTED VERSION"""
    
    def __init__(self, model: nn.Module, config: Optional[SEALConfig] = None,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.model = model
        self.config = config or SEALConfig()
        self.device = device or next(model.parameters()).device
        self._saved_state = None
        self._original_state = deepcopy(model.state_dict())
        self.consistency_checker = ConsistencyChecker(ssim_threshold=self.config.consistency_threshold)
        
        if GaussianDiffusion:
            self.diffusion = GaussianDiffusion(T=self.config.max_diffusion_steps,
                                               schedule=self.config.diffusion_schedule, device=str(self.device))
        else:
            self.diffusion = None
        
        self._ttt_optimizer = None
        self._ema_pseudo_label = None
        self.adaptation_stats = {'total_samples': 0, 'accepted_samples': 0, 'rejected_samples': 0,
                                  'total_updates': 0, 'avg_confidence': 0.0}
    
    def _get_adaptable_parameters(self) -> List[nn.Parameter]:
        if self.config.adapt_layers == 'all':
            return list(self.model.parameters())
        elif self.config.adapt_layers == 'attention':
            params = []
            for name, m in self.model.named_modules():
                if 'attn' in name.lower():
                    params.extend(m.parameters())
            return params
        elif self.config.adapt_layers == 'final':
            return list(self.model.final_layer.parameters()) if hasattr(self.model, 'final_layer') else []
        return list(self.model.parameters())
    
    def _create_ttt_optimizer(self):
        params = self._get_adaptable_parameters()
        return SGD(params, lr=self.config.ttt_lr, momentum=self.config.momentum if self.config.use_momentum else 0.0,
                  weight_decay=self.config.weight_decay)
    
    def save_state(self):
        self._saved_state = deepcopy(self.model.state_dict())
    
    def restore_state(self):
        if self._saved_state:
            self.model.load_state_dict(self._saved_state)
            self._saved_state = None
            self._ttt_optimizer = None
    
    def reset_to_original(self):
        if self._original_state:
            self.model.load_state_dict(self._original_state)
        self._saved_state = None
        self._ttt_optimizer = None
        self._ema_pseudo_label = None
        self.adaptation_stats = {'total_samples': 0, 'accepted_samples': 0, 'rejected_samples': 0,
                                  'total_updates': 0, 'avg_confidence': 0.0}
    
    def _move_diffusion_to_device(self):
        if not self.diffusion:
            return
        for attr in ['beta', 'alpha', 'alphabar', 'sqrt_alphabar', 'sqrt_one_minus_alphabar',
                     'sqrt_recip_alphabar', 'posterior_variance', 'posterior_log_variance',
                     'posterior_mean_coef1', 'posterior_mean_coef2']:
            if hasattr(self.diffusion, attr):
                setattr(self.diffusion, attr, getattr(self.diffusion, attr).to(self.device))
    
    @torch.no_grad()
    def generate_self_edit(self, input_scan: torch.Tensor, intv_t=None, treat_code=None, i_tg=None,
                           num_consistency_samples: int = 2) -> Tuple[Optional[torch.Tensor], Dict]:
        self.model.eval()
        input_scan = input_scan.to(self.device)
        B, C, H, W = input_scan.shape
        
        if i_tg is None:
            i_tg = -torch.ones((B,), dtype=torch.long, device=self.device)
        
        self._move_diffusion_to_device()
        predictions = []
        
        for _ in range(num_consistency_samples):
            if self.diffusion:
                pred, _ = self.diffusion.ddim_inverse(net=self.model, x=input_scan.clone(), intv=intv_t,
                                                       treat_cond=treat_code, i_tg=i_tg,
                                                       steps=self.config.max_diffusion_steps, eta=0.0, device=self.device)
            else:
                t = torch.ones((B,), device=self.device) * 10
                pred = self.model(input_scan, t, intv_t=intv_t, treat_code=treat_code, i_tg=i_tg)[:, 4:7]
            predictions.append(pred)
        
        main_pred = predictions[0]
        i_tg_pos = torch.where(i_tg < 0, 4 + i_tg, i_tg).long()
        target_imgs = torch.stack([input_scan.view(B, 4, 3, H, W)[i, i_tg_pos[i].item()] for i in range(B)], dim=0)
        
        is_consistent, metrics = self.consistency_checker.check_consistency(
            target_imgs, main_pred, predictions[1:] if len(predictions) > 1 else None)
        
        confidence = 1.0 / (1.0 + metrics.get('variance', 0.0) * self.config.temperature)
        n = self.adaptation_stats['total_samples'] + 1
        self.adaptation_stats['avg_confidence'] += (confidence - self.adaptation_stats['avg_confidence']) / n
        
        info = {'accepted': False, 'metrics': metrics, 'confidence': confidence}
        
        if is_consistent and confidence > self.config.confidence_threshold:
            info['accepted'] = True
            pseudo_target = main_pred
            if self.config.ema_decay > 0 and self._ema_pseudo_label is not None:
                pseudo_target = self.config.ema_decay * self._ema_pseudo_label + (1 - self.config.ema_decay) * main_pred
            self._ema_pseudo_label = pseudo_target.clone()
            self.adaptation_stats['accepted_samples'] += 1
            return pseudo_target, info
        
        self.adaptation_stats['rejected_samples'] += 1
        return None, info
    
    def apply_update(self, input_scan: torch.Tensor, pseudo_target: torch.Tensor, intv_t=None,
                     treat_code=None, i_tg=None, num_steps=None) -> Dict[str, float]:
        self.model.train()
        num_steps = num_steps or self.config.ttt_steps
        
        if self.config.reset_optimizer_per_sample or self._ttt_optimizer is None:
            self._ttt_optimizer = self._create_ttt_optimizer()
        
        input_scan = input_scan.to(self.device)
        pseudo_target = pseudo_target.to(self.device).detach()
        B, C, H, W = input_scan.shape
        
        if i_tg is None:
            i_tg = -torch.ones((B,), dtype=torch.long, device=self.device)
        i_tg_pos = torch.where(i_tg < 0, 4 + i_tg, i_tg).long()
        
        self._move_diffusion_to_device()
        losses = {}
        
        for step in range(num_steps):
            lr = self.config.ttt_lr
            if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
                lr = self.config.ttt_lr * (step + 1) / self.config.warmup_steps
            elif self.config.use_cosine_schedule and num_steps > 1:
                progress = (step - self.config.warmup_steps) / max(1, num_steps - self.config.warmup_steps)
                lr = self.config.ttt_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            for pg in self._ttt_optimizer.param_groups:
                pg['lr'] = lr
            
            self._ttt_optimizer.zero_grad()
            t = torch.randint(1, self.config.max_diffusion_steps + 1, (B,), device=self.device)
            
            if self.diffusion:
                noisy_target, epsilon = self.diffusion.sample(pseudo_target, t)
            else:
                epsilon = torch.randn_like(pseudo_target)
                noisy_target = pseudo_target + (t.float().view(B,1,1,1) / self.config.max_diffusion_steps) * epsilon
            
            input_mod = input_scan.view(B, 4, 3, H, W).clone()
            for i in range(B):
                input_mod[i, i_tg_pos[i].item()] = noisy_target[i]
            
            output = self.model(input_mod.view(B, 12, H, W), t.float(), intv_t=intv_t, treat_code=treat_code, i_tg=i_tg)
            loss = F.mse_loss(output[:, 4:7], epsilon)
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._get_adaptable_parameters(), self.config.gradient_clip)
            
            self._ttt_optimizer.step()
            losses[f'step_{step}'] = loss.item()
            self.adaptation_stats['total_updates'] += 1
        
        self.model.eval()
        return losses
    
    def adapt_single_sample(self, input_scan: torch.Tensor, intv_t=None, treat_code=None,
                            i_tg=None, persistent: bool = False) -> Dict[str, Any]:
        self.adaptation_stats['total_samples'] += 1
        result = {'adapted': False, 'pseudo_label_generated': False, 'update_applied': False, 'metrics': {}, 'losses': {}}
        
        if not persistent:
            self.save_state()
        
        try:
            pseudo_target, gen_info = self.generate_self_edit(input_scan, intv_t, treat_code, i_tg)
            result['pseudo_label_generated'] = gen_info['accepted']
            result['metrics'] = gen_info.get('metrics', {})
            result['confidence'] = gen_info.get('confidence', 0.0)
            
            if pseudo_target is not None:
                result['losses'] = self.apply_update(input_scan, pseudo_target, intv_t, treat_code, i_tg)
                result['update_applied'] = True
                result['adapted'] = True
        except Exception as e:
            result['error'] = str(e)
            if not persistent and self._saved_state:
                self.restore_state()
        
        return result
    
    @torch.no_grad()
    def infer(self, input_scan: torch.Tensor, intv_t=None, treat_code=None, i_tg=None,
              diffusion_steps=None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        input_scan = input_scan.to(self.device)
        B = input_scan.shape[0]
        
        if i_tg is None:
            i_tg = -torch.ones((B,), dtype=torch.long, device=self.device)
        
        self._move_diffusion_to_device()
        steps = diffusion_steps or self.config.max_diffusion_steps
        
        if self.diffusion:
            return self.diffusion.ddim_inverse(net=self.model, x=input_scan, intv=intv_t, treat_cond=treat_code,
                                                i_tg=i_tg, steps=steps, eta=0.0, device=self.device)
        else:
            output = self.model(input_scan, torch.ones((B,), device=self.device), intv_t=intv_t,
                               treat_code=treat_code, i_tg=i_tg)
            return output[:, 4:7], output[:, 0:4]
    
    def adapt_and_infer(self, input_scan: torch.Tensor, intv_t=None, treat_code=None, i_tg=None,
                        persistent=False, rollback_after=True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        result = self.adapt_single_sample(input_scan, intv_t, treat_code, i_tg, persistent)
        pred_img, pred_mask = self.infer(input_scan, intv_t, treat_code, i_tg)
        if rollback_after and not persistent and self._saved_state:
            self.restore_state()
        return pred_img, pred_mask, result
    
    def get_stats(self) -> Dict:
        return self.adaptation_stats.copy()
    
    def forward(self, x, timesteps, intv_t=None, treat_code=None, i_tg=None):
        return self.model(x, timesteps, intv_t=intv_t, treat_code=treat_code, i_tg=i_tg)


def create_seal_adapter(model, ttt_lr=1e-4, ttt_steps=1, confidence_threshold=0.5,
                        consistency_threshold=0.7, adapt_layers='all', max_diffusion_steps=50, device=None):
    config = SEALConfig(ttt_lr=ttt_lr, ttt_steps=ttt_steps, confidence_threshold=confidence_threshold,
                        consistency_threshold=consistency_threshold, adapt_layers=adapt_layers,
                        max_diffusion_steps=max_diffusion_steps)
    return SEALAdapter(model=model, config=config, device=device)
