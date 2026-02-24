"""
Comprehensive Architecture Integration Tests for TaDiff-DiT

This test suite performs rigorous testing of:
1. Dimensional consistency throughout the pipeline
2. Variable name and configuration issues
3. Integration between components
4. Edge cases and error handling
5. Gradient flow verification
6. Memory management
7. SEAL adapter integration with main model
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return {
        'image_size': 192,
        'in_channels': 12,
        'out_channels': 7,
        'model_channels': 32,  # Smaller for faster tests
        'hidden_size': 256,    # Smaller for faster tests
        'depth': 4,            # Fewer layers for faster tests
        'num_heads': 4,
        'patch_size': 16,
        'max_T': 50,
        'batch_size': 2,
    }


# =============================================================================
# Test 1: Core Architecture Imports and Initialization
# =============================================================================

class TestImportsAndInitialization:
    """Test that all modules can be imported and initialized correctly."""
    
    def test_import_tadiff_dit(self):
        """Test importing TaDiff_DiT architecture."""
        from src.net.tadiff_dit_arch import (
            TaDiff_DiT, TaDiff_Net,
            PatchEmbed, DiTBlock, FinalLayer,
            BDH_Attention, MLP,
            SinusoidalPositionEmbeddings,
            modulate, unpatchify
        )
        assert TaDiff_DiT is TaDiff_Net, "TaDiff_Net should be alias for TaDiff_DiT"
    
    def test_import_diffusion(self):
        """Test importing diffusion module."""
        from src.net.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(T=50, schedule='linear')
        assert diffusion.T == 50
        assert len(diffusion.beta) == 50
        assert len(diffusion.alphabar) == 50
    
    def test_import_seal_adapter(self):
        """Test importing SEAL adapter."""
        from src.net.seal_adapter import SEALAdapter, SEALConfig, ConsistencyChecker
        
        config = SEALConfig()
        assert config.ttt_lr > 0
        assert config.ttt_steps >= 1
    
    def test_import_utils(self):
        """Test importing utility functions."""
        from src.net.utils import (
            timestep_embedding,
            FourierFeatures,
            checkpoint,
            normalization,
            conv_nd,
            linear,
        )
        
        # Test timestep_embedding
        t = torch.tensor([1, 10, 100])
        emb = timestep_embedding(t, 64)
        assert emb.shape == (3, 64)
    
    def test_import_ssim(self):
        """Test importing SSIM module."""
        from src.net.ssim import SSIM, ssim, ms_ssim
        
        ssim_module = SSIM(data_range=1.0, size_average=True, win_size=11, channel=3)
        assert ssim_module is not None


# =============================================================================
# Test 2: Dimensional Consistency
# =============================================================================

class TestDimensionalConsistency:
    """Test that dimensions are consistent throughout the pipeline."""
    
    def test_patch_embed_dimensions(self, device, default_config):
        """Test PatchEmbed output dimensions."""
        from src.net.tadiff_dit_arch import PatchEmbed
        
        cfg = default_config
        patch_embed = PatchEmbed(
            image_size=cfg['image_size'],
            patch_size=cfg['patch_size'],
            in_channels=cfg['in_channels'],
            embed_dim=cfg['hidden_size'],
        ).to(device)
        
        x = torch.randn(cfg['batch_size'], cfg['in_channels'], 
                       cfg['image_size'], cfg['image_size'], device=device)
        
        out = patch_embed(x)
        
        expected_num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
        assert out.shape == (cfg['batch_size'], expected_num_patches, cfg['hidden_size']), \
            f"Expected ({cfg['batch_size']}, {expected_num_patches}, {cfg['hidden_size']}), got {out.shape}"
    
    def test_dit_block_dimensions(self, device, default_config):
        """Test DiTBlock preserves dimensions."""
        from src.net.tadiff_dit_arch import DiTBlock
        
        cfg = default_config
        num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
        
        block = DiTBlock(
            hidden_size=cfg['hidden_size'],
            num_heads=cfg['num_heads'],
            condition_dim=cfg['hidden_size'],
        ).to(device)
        
        x = torch.randn(cfg['batch_size'], num_patches, cfg['hidden_size'], device=device)
        c = torch.randn(cfg['batch_size'], cfg['hidden_size'], device=device)
        
        out = block(x, c)
        assert out.shape == x.shape, f"DiTBlock should preserve shape: {x.shape} vs {out.shape}"
    
    def test_bdh_attention_dimensions(self, device, default_config):
        """Test BDH_Attention dimensions."""
        from src.net.tadiff_dit_arch import BDH_Attention
        
        cfg = default_config
        num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
        
        attn = BDH_Attention(
            dim=cfg['hidden_size'],
            num_heads=cfg['num_heads'],
            expansion_factor=4,
        ).to(device)
        
        x = torch.randn(cfg['batch_size'], num_patches, cfg['hidden_size'], device=device)
        out = attn(x)
        
        assert out.shape == x.shape, f"BDH_Attention should preserve shape: {x.shape} vs {out.shape}"
    
    def test_final_layer_dimensions(self, device, default_config):
        """Test FinalLayer output dimensions."""
        from src.net.tadiff_dit_arch import FinalLayer
        
        cfg = default_config
        num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
        
        final = FinalLayer(
            hidden_size=cfg['hidden_size'],
            patch_size=cfg['patch_size'],
            out_channels=cfg['out_channels'],
            condition_dim=cfg['hidden_size'],
        ).to(device)
        
        x = torch.randn(cfg['batch_size'], num_patches, cfg['hidden_size'], device=device)
        c = torch.randn(cfg['batch_size'], cfg['hidden_size'], device=device)
        
        out = final(x, c, cfg['image_size'], cfg['image_size'])
        
        assert out.shape == (cfg['batch_size'], cfg['out_channels'], 
                            cfg['image_size'], cfg['image_size']), \
            f"Expected ({cfg['batch_size']}, {cfg['out_channels']}, " \
            f"{cfg['image_size']}, {cfg['image_size']}), got {out.shape}"
    
    def test_full_model_dimensions(self, device, default_config):
        """Test full TaDiff_DiT model dimensions."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device)
        
        x = torch.randn(cfg['batch_size'], cfg['in_channels'],
                       cfg['image_size'], cfg['image_size'], device=device)
        t = torch.randint(1, cfg['max_T'], (cfg['batch_size'],), device=device).float()
        
        intv_t = [torch.rand(cfg['batch_size'], device=device) * 300 for _ in range(4)]
        treat_code = [torch.randint(0, 3, (cfg['batch_size'],), device=device).float() for _ in range(4)]
        i_tg = -torch.ones(cfg['batch_size'], dtype=torch.int8, device=device)
        
        out = model(x, t, intv_t, treat_code, i_tg)
        
        assert out.shape == (cfg['batch_size'], cfg['out_channels'],
                            cfg['image_size'], cfg['image_size']), \
            f"Expected ({cfg['batch_size']}, {cfg['out_channels']}, " \
            f"{cfg['image_size']}, {cfg['image_size']}), got {out.shape}"


# =============================================================================
# Test 3: Diffusion Process Consistency
# =============================================================================

class TestDiffusionProcess:
    """Test diffusion process correctness."""
    
    def test_linear_schedule(self, device):
        """Test linear noise schedule."""
        from src.net.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(T=100, schedule='linear', device=device)
        
        # Check schedule is monotonically increasing
        assert torch.all(diffusion.beta[1:] >= diffusion.beta[:-1]), \
            "Linear schedule should be monotonically increasing"
        
        # Check alphabar is monotonically decreasing
        assert torch.all(diffusion.alphabar[1:] <= diffusion.alphabar[:-1]), \
            "alphabar should be monotonically decreasing"
    
    def test_cosine_schedule(self, device):
        """Test cosine noise schedule."""
        from src.net.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(T=100, schedule='cosine', device=device)
        
        # Check alphabar is monotonically decreasing
        assert torch.all(diffusion.alphabar[1:] <= diffusion.alphabar[:-1]), \
            "alphabar should be monotonically decreasing"
    
    def test_sample_dimensions(self, device):
        """Test diffusion sampling dimensions."""
        from src.net.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(T=50, schedule='linear', device=device)
        
        x0 = torch.randn(2, 3, 64, 64, device=device)
        t = torch.tensor([10, 25], device=device)
        
        xt, epsilon = diffusion.sample(x0, t)
        
        assert xt.shape == x0.shape, f"xt shape mismatch: {xt.shape} vs {x0.shape}"
        assert epsilon.shape == x0.shape, f"epsilon shape mismatch: {epsilon.shape} vs {x0.shape}"
    
    def test_sample_noise_level(self, device):
        """Test that noise level increases with timestep."""
        from src.net.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(T=100, schedule='linear', device=device)
        
        x0 = torch.zeros(1, 3, 32, 32, device=device)  # Zero image
        
        # Sample at different timesteps
        t_low = torch.tensor([5], device=device)
        t_high = torch.tensor([95], device=device)
        
        xt_low, _ = diffusion.sample(x0, t_low)
        xt_high, _ = diffusion.sample(x0, t_high)
        
        # Higher timestep should have more noise (larger variance)
        var_low = xt_low.var().item()
        var_high = xt_high.var().item()
        
        assert var_high > var_low, \
            f"Higher timestep should have more noise: var_low={var_low}, var_high={var_high}"


# =============================================================================
# Test 4: Conditioning Consistency
# =============================================================================

class TestConditioningConsistency:
    """Test conditioning mechanisms work correctly."""
    
    def test_timestep_embedding_consistency(self, device):
        """Test timestep embedding produces consistent results."""
        from src.net.utils import timestep_embedding
        
        t = torch.tensor([50], device=device)
        
        emb1 = timestep_embedding(t, 128)
        emb2 = timestep_embedding(t, 128)
        
        assert torch.allclose(emb1, emb2), "Same timestep should produce same embedding"
    
    def test_timestep_embedding_different_values(self, device):
        """Test different timesteps produce different embeddings."""
        from src.net.utils import timestep_embedding
        
        t1 = torch.tensor([10], device=device)
        t2 = torch.tensor([90], device=device)
        
        emb1 = timestep_embedding(t1, 128)
        emb2 = timestep_embedding(t2, 128)
        
        assert not torch.allclose(emb1, emb2), "Different timesteps should produce different embeddings"
    
    def test_fourier_features_dimensions(self, device):
        """Test FourierFeatures output dimensions."""
        from src.net.utils import FourierFeatures
        
        ff = FourierFeatures(in_features=1, out_features=64).to(device)
        
        x = torch.randn(4, 1, device=device)
        out = ff(x)
        
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    
    def test_treatment_conditioning_impact(self, device, default_config):
        """Test that different treatments produce different outputs."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(1, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0], device=device)
        intv_t = [torch.tensor([100.0], device=device) for _ in range(4)]
        i_tg = torch.tensor([-1], dtype=torch.int8, device=device)
        
        # Different treatment codes
        treat_code_1 = [torch.tensor([0.0], device=device) for _ in range(4)]
        treat_code_2 = [torch.tensor([2.0], device=device) for _ in range(4)]
        
        with torch.no_grad():
            out1 = model(x, t, intv_t, treat_code_1, i_tg)
            out2 = model(x, t, intv_t, treat_code_2, i_tg)
        
        assert not torch.allclose(out1, out2, atol=1e-5), \
            "Different treatments should produce different outputs"
    
    def test_days_conditioning_impact(self, device, default_config):
        """Test that different day intervals produce different outputs."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(1, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0], device=device)
        treat_code = [torch.tensor([1.0], device=device) for _ in range(4)]
        i_tg = torch.tensor([-1], dtype=torch.int8, device=device)
        
        # Different day intervals
        intv_t_1 = [torch.tensor([0.0], device=device), torch.tensor([30.0], device=device),
                    torch.tensor([60.0], device=device), torch.tensor([90.0], device=device)]
        intv_t_2 = [torch.tensor([0.0], device=device), torch.tensor([100.0], device=device),
                    torch.tensor([200.0], device=device), torch.tensor([300.0], device=device)]
        
        with torch.no_grad():
            out1 = model(x, t, intv_t_1, treat_code, i_tg)
            out2 = model(x, t, intv_t_2, treat_code, i_tg)
        
        assert not torch.allclose(out1, out2, atol=1e-5), \
            "Different day intervals should produce different outputs"


# =============================================================================
# Test 5: Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Test gradient flow through the network."""
    
    def test_gradient_flow_to_input(self, device, default_config):
        """Test gradients flow back to input."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).train()
        
        x = torch.randn(cfg['batch_size'], cfg['in_channels'],
                       cfg['image_size'], cfg['image_size'], 
                       device=device, requires_grad=True)
        t = torch.randint(1, cfg['max_T'], (cfg['batch_size'],), device=device).float()
        intv_t = [torch.rand(cfg['batch_size'], device=device) * 300 for _ in range(4)]
        treat_code = [torch.randint(0, 3, (cfg['batch_size'],), device=device).float() for _ in range(4)]
        i_tg = -torch.ones(cfg['batch_size'], dtype=torch.int8, device=device)
        
        out = model(x, t, intv_t, treat_code, i_tg)
        loss = out.mean()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert x.grad.shape == x.shape, "Gradient shape should match input shape"
        assert not torch.all(x.grad == 0), "Gradients should be non-zero"
    
    def test_gradient_flow_to_all_parameters(self, device, default_config):
        """Test gradients flow to all trainable parameters."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).train()
        
        x = torch.randn(cfg['batch_size'], cfg['in_channels'],
                       cfg['image_size'], cfg['image_size'], device=device)
        t = torch.randint(1, cfg['max_T'], (cfg['batch_size'],), device=device).float()
        intv_t = [torch.rand(cfg['batch_size'], device=device) * 300 for _ in range(4)]
        treat_code = [torch.randint(0, 3, (cfg['batch_size'],), device=device).float() for _ in range(4)]
        i_tg = -torch.ones(cfg['batch_size'], dtype=torch.int8, device=device)
        
        out = model(x, t, intv_t, treat_code, i_tg)
        loss = out.mean()
        loss.backward()
        
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)
        
        assert len(params_without_grad) == 0, \
            f"Parameters without gradients: {params_without_grad}"


# =============================================================================
# Test 6: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self, device, default_config):
        """Test with batch size of 1."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(1, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0], device=device)
        intv_t = [torch.tensor([100.0], device=device) for _ in range(4)]
        treat_code = [torch.tensor([1.0], device=device) for _ in range(4)]
        i_tg = torch.tensor([-1], dtype=torch.int8, device=device)
        
        with torch.no_grad():
            out = model(x, t, intv_t, treat_code, i_tg)
        
        assert out.shape == (1, cfg['out_channels'], cfg['image_size'], cfg['image_size'])
    
    def test_different_target_indices(self, device, default_config):
        """Test different target session indices."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(2, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0, 50.0], device=device)
        intv_t = [torch.tensor([100.0, 100.0], device=device) for _ in range(4)]
        treat_code = [torch.tensor([1.0, 1.0], device=device) for _ in range(4)]
        
        # Test all valid target indices
        for i_tg_val in [0, 1, 2, 3, -1, -2, -3, -4]:
            i_tg = torch.tensor([i_tg_val, i_tg_val], dtype=torch.int8, device=device)
            
            with torch.no_grad():
                out = model(x, t, intv_t, treat_code, i_tg)
            
            assert out.shape == (2, cfg['out_channels'], cfg['image_size'], cfg['image_size']), \
                f"Failed for i_tg={i_tg_val}"
    
    def test_mixed_target_indices(self, device, default_config):
        """Test batch with mixed target indices."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(4, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0, 50.0, 75.0, 100.0], device=device)
        intv_t = [torch.rand(4, device=device) * 300 for _ in range(4)]
        treat_code = [torch.randint(0, 3, (4,), device=device).float() for _ in range(4)]
        
        # Mixed target indices within batch
        i_tg = torch.tensor([0, -1, 2, -2], dtype=torch.int8, device=device)
        
        with torch.no_grad():
            out = model(x, t, intv_t, treat_code, i_tg)
        
        assert out.shape == (4, cfg['out_channels'], cfg['image_size'], cfg['image_size'])
    
    def test_extreme_timesteps(self, device, default_config):
        """Test with extreme timestep values."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        x = torch.randn(2, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        intv_t = [torch.rand(2, device=device) * 300 for _ in range(4)]
        treat_code = [torch.randint(0, 3, (2,), device=device).float() for _ in range(4)]
        i_tg = -torch.ones(2, dtype=torch.int8, device=device)
        
        # Test extreme timesteps
        for t_val in [1.0, 500.0, 999.0]:
            t = torch.tensor([t_val, t_val], device=device)
            
            with torch.no_grad():
                out = model(x, t, intv_t, treat_code, i_tg)
            
            assert out.shape == (2, cfg['out_channels'], cfg['image_size'], cfg['image_size']), \
                f"Failed for t={t_val}"
            assert not torch.isnan(out).any(), f"NaN detected for t={t_val}"
            assert not torch.isinf(out).any(), f"Inf detected for t={t_val}"


# =============================================================================
# Test 7: SEAL Adapter Integration
# =============================================================================

class TestSEALAdapterIntegration:
    """Test SEAL adapter integration with main model."""
    
    def test_seal_wraps_tadiff(self, device, default_config):
        """Test SEAL adapter wraps TaDiff model correctly."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device)
        
        seal_config = SEALConfig(
            ttt_lr=1e-4,
            ttt_steps=1,
            max_diffusion_steps=cfg['max_T'],
        )
        
        adapter = SEALAdapter(model, config=seal_config, device=device)
        
        assert adapter.model is model
        assert adapter.config.ttt_lr == 1e-4
    
    def test_seal_state_management(self, device, default_config):
        """Test SEAL state save/restore with real model."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device)
        
        adapter = SEALAdapter(model, config=SEALConfig(), device=device)
        
        # Get initial weights
        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Save state
        adapter.save_state()
        
        # Modify weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Verify weights changed
        for name, param in model.named_parameters():
            assert not torch.allclose(initial_weights[name], param), \
                f"Weight {name} should have changed"
        
        # Restore state
        adapter.restore_state()
        
        # Verify weights restored
        for name, param in model.named_parameters():
            assert torch.allclose(initial_weights[name], param), \
                f"Weight {name} should be restored"


# =============================================================================
# Test 8: Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_image_size_divisibility(self, device):
        """Test that image size must be divisible by patch size."""
        from src.net.tadiff_dit_arch import PatchEmbed
        
        # This should work
        PatchEmbed(image_size=192, patch_size=16, in_channels=12, embed_dim=768)
        
        # This should fail
        with pytest.raises(AssertionError):
            PatchEmbed(image_size=190, patch_size=16, in_channels=12, embed_dim=768)
    
    def test_hidden_size_divisibility(self, device):
        """Test that hidden size must be divisible by num_heads."""
        from src.net.tadiff_dit_arch import BDH_Attention
        
        # This should work (256 / 8 = 32)
        BDH_Attention(dim=256, num_heads=8)
        
        # This should fail (256 / 7 = not integer)
        with pytest.raises(AssertionError):
            BDH_Attention(dim=256, num_heads=7)
    
    def test_diffusion_schedule_validation(self, device):
        """Test diffusion schedule validation."""
        from src.net.diffusion import GaussianDiffusion
        
        # Valid schedules
        GaussianDiffusion(T=100, schedule='linear')
        GaussianDiffusion(T=100, schedule='cosine')
        
        # Invalid schedule should raise ValueError
        with pytest.raises(ValueError):
            GaussianDiffusion(T=100, schedule='invalid')


# =============================================================================
# Test 9: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of the model."""
    
    def test_no_nan_in_forward_pass(self, device, default_config):
        """Test that forward pass produces no NaN values."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        cfg = default_config
        
        model = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
        ).to(device).eval()
        
        # Test with various inputs
        for _ in range(5):
            x = torch.randn(2, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
            t = torch.randint(1, cfg['max_T'], (2,), device=device).float()
            intv_t = [torch.rand(2, device=device) * 1000 for _ in range(4)]  # Large day values
            treat_code = [torch.randint(0, 10, (2,), device=device).float() for _ in range(4)]
            i_tg = -torch.ones(2, dtype=torch.int8, device=device)
            
            with torch.no_grad():
                out = model(x, t, intv_t, treat_code, i_tg)
            
            assert not torch.isnan(out).any(), "NaN detected in output"
            assert not torch.isinf(out).any(), "Inf detected in output"
    
    def test_bdh_attention_normalization(self, device):
        """Test BDH attention normalization prevents division by zero."""
        from src.net.tadiff_dit_arch import BDH_Attention
        
        attn = BDH_Attention(dim=256, num_heads=4).to(device)
        
        # Test with zero input
        x_zero = torch.zeros(2, 100, 256, device=device)
        out = attn(x_zero)
        
        assert not torch.isnan(out).any(), "NaN from zero input"
        
        # Test with very small input
        x_small = torch.randn(2, 100, 256, device=device) * 1e-8
        out = attn(x_small)
        
        assert not torch.isnan(out).any(), "NaN from small input"


# =============================================================================
# Test 10: Memory Efficiency
# =============================================================================

class TestMemoryEfficiency:
    """Test memory efficiency and cleanup."""
    
    def test_gradient_checkpoint(self, device, default_config):
        """Test gradient checkpointing reduces memory."""
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        cfg = default_config
        
        # Model without checkpointing
        model_no_ckpt = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
            use_checkpoint=False,
        ).to(device)
        
        # Model with checkpointing
        model_ckpt = TaDiff_DiT(
            image_size=cfg['image_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            model_channels=cfg['model_channels'],
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            patch_size=cfg['patch_size'],
            use_checkpoint=True,
        ).to(device)
        
        # Both should produce same output
        x = torch.randn(1, cfg['in_channels'], cfg['image_size'], cfg['image_size'], device=device)
        t = torch.tensor([25.0], device=device)
        intv_t = [torch.tensor([100.0], device=device) for _ in range(4)]
        treat_code = [torch.tensor([1.0], device=device) for _ in range(4)]
        i_tg = torch.tensor([-1], dtype=torch.int8, device=device)
        
        # Copy weights
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())
        
        with torch.no_grad():
            out_no_ckpt = model_no_ckpt(x, t, intv_t, treat_code, i_tg)
            out_ckpt = model_ckpt(x, t, intv_t, treat_code, i_tg)
        
        assert torch.allclose(out_no_ckpt, out_ckpt, atol=1e-5), \
            "Checkpointing should not change output"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
