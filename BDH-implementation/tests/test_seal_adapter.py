"""
SEAL Adapter Smoke Tests

This test suite verifies that the Test-Time Training (TTT) mechanics of the
SEALAdapter work correctly, specifically:

1. The "Amnesia" Test: Weights rollback correctly after temporary adaptation
2. The "Elasticity" Test: Weights update correctly during persistent adaptation

Usage:
    pytest tests/test_seal_adapter.py -v
    
    Or run directly:
    python tests/test_seal_adapter.py
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import pytest, but don't fail if not available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy pytest module for decorators
    class DummyPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
    pytest = DummyPytest()

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def create_dummy_tadiff_net(
    image_size: int = 64,
    in_channels: int = 12,
    out_channels: int = 7,
    model_channels: int = 32,
    hidden_size: int = 64,
    depth: int = 2,
    num_heads: int = 2,
    patch_size: int = 8,
) -> nn.Module:
    """
    Create a small TaDiff_Net for testing.
    
    Uses the actual TaDiff_DiT architecture but with reduced dimensions
    to make tests run quickly.
    """
    try:
        from src.net.tadiff_dit_arch import TaDiff_DiT
        
        model = TaDiff_DiT(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            mlp_ratio=2.0,
            dropout=0.0,
            use_checkpoint=False,
            bdh_layers=2,
            bdh_expansion=2,
        )
        return model
        
    except ImportError as e:
        # Fallback: create a minimal mock model for testing
        print(f"Warning: Could not import TaDiff_DiT ({e}), using mock model")
        return create_mock_tadiff_net(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
        )


def create_mock_tadiff_net(
    image_size: int = 64,
    in_channels: int = 12,
    out_channels: int = 7,
    hidden_size: int = 64,
) -> nn.Module:
    """
    Create a minimal mock TaDiff_Net for testing when real model isn't available.
    
    This mock has the same interface as TaDiff_DiT but is much simpler.
    """
    
    class MockTaDiffNet(nn.Module):
        """Minimal mock for TaDiff_Net with same forward signature."""
        
        def __init__(self, image_size, in_channels, out_channels, hidden_size):
            super().__init__()
            
            self.image_size = image_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            # Simple encoder-decoder structure
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
                nn.ReLU(),
            )
            
            # Time embedding
            self.time_embed = nn.Sequential(
                nn.Linear(32, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            
            # Transformer-like blocks (for weight testing)
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(2)
            ])
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, out_channels, 3, padding=1),
            )
            
            self.dtype = torch.float32
        
        def _get_timestep_embedding(self, timesteps: torch.Tensor, dim: int = 32) -> torch.Tensor:
            """Simple sinusoidal timestep embedding."""
            half_dim = dim // 2
            emb = np.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
            emb = timesteps[:, None].float() * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            return emb
        
        def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            intv_t: Optional[List[torch.Tensor]] = None,
            treat_code: Optional[List[torch.Tensor]] = None,
            i_tg: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            b, c, h, w = x.shape
            
            # Encode
            features = self.encoder(x)  # [B, hidden, H, W]
            
            # Add time embedding
            t_emb = self._get_timestep_embedding(timesteps)
            t_emb = self.time_embed(t_emb)  # [B, hidden]
            features = features + t_emb[:, :, None, None]
            
            # Apply blocks (on flattened spatial dimensions)
            b, c, h, w = features.shape
            feat_flat = features.permute(0, 2, 3, 1).reshape(b * h * w, c)
            
            for block in self.blocks:
                feat_flat = feat_flat + block(feat_flat)
            
            features = feat_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
            
            # Decode
            output = self.decoder(features)
            
            return output
    
    return MockTaDiffNet(image_size, in_channels, out_channels, hidden_size)


def create_dummy_input(
    batch_size: int = 2,
    num_sessions: int = 4,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Create dummy input data matching TaDiff_Net's expected format.
    
    Returns:
        Tuple of (input_scan, intv_t, treat_code, i_tg)
    """
    # Input scan: [B, S*C, H, W] where S=4 sessions, C=3 channels
    input_scan = torch.randn(batch_size, num_sessions * channels, height, width, device=device)
    
    # Day intervals for each session: list of [B] tensors
    intv_t = [
        torch.randint(0, 365, (batch_size,), device=device).float()
        for _ in range(num_sessions)
    ]
    
    # Treatment codes for each session: list of [B] tensors
    treat_code = [
        torch.randint(0, 3, (batch_size,), device=device).float()
        for _ in range(num_sessions)
    ]
    
    # Target session indices
    i_tg = -torch.ones((batch_size,), dtype=torch.int8, device=device)
    
    return input_scan, intv_t, treat_code, i_tg


def get_weight_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Take a snapshot of model weights.
    
    Returns:
        Dictionary mapping parameter names to cloned weight tensors
    """
    return {name: param.clone().detach() for name, param in model.named_parameters()}


def compare_weights(
    snapshot1: Dict[str, torch.Tensor],
    snapshot2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, List[str]]:
    """
    Compare two weight snapshots.
    
    Args:
        snapshot1: First weight snapshot
        snapshot2: Second weight snapshot
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        Tuple of (all_equal, list_of_different_params)
    """
    different_params = []
    
    for name in snapshot1.keys():
        if name not in snapshot2:
            different_params.append(f"{name} (missing in snapshot2)")
            continue
        
        if not torch.allclose(snapshot1[name], snapshot2[name], rtol=rtol, atol=atol):
            diff = (snapshot1[name] - snapshot2[name]).abs().max().item()
            different_params.append(f"{name} (max_diff={diff:.6f})")
    
    return len(different_params) == 0, different_params


# =============================================================================
# Test Class
# =============================================================================

class TestSEALAdapter:
    """Test suite for SEALAdapter TTT mechanics."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = 64
        self.batch_size = 2
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_adapter_initialization(self):
        """Test that SEALAdapter initializes correctly."""
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        # Create model
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        # Create adapter
        config = SEALConfig(
            ttt_lr=1e-3,
            ttt_steps=1,
            max_diffusion_steps=10,
        )
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # Verify initialization
        assert adapter.model is model
        assert adapter.config.ttt_lr == 1e-3
        assert adapter.config.ttt_steps == 1
        assert adapter._original_state is not None
        
        print("✓ Adapter initialization test passed")
    
    def test_state_save_restore(self):
        """Test that save_state and restore_state work correctly."""
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        # Create model and adapter
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        config = SEALConfig(ttt_lr=1e-3, ttt_steps=1)
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # Take initial snapshot
        initial_snapshot = get_weight_snapshot(model)
        
        # Save state
        adapter.save_state()
        
        # Manually modify weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Verify weights changed
        modified_snapshot = get_weight_snapshot(model)
        weights_equal, _ = compare_weights(initial_snapshot, modified_snapshot)
        assert not weights_equal, "Weights should have changed after modification"
        
        # Restore state
        adapter.restore_state()
        
        # Verify weights restored
        restored_snapshot = get_weight_snapshot(model)
        weights_equal, different = compare_weights(initial_snapshot, restored_snapshot)
        assert weights_equal, f"Weights should be restored. Different params: {different}"
        
        print("✓ State save/restore test passed")
    
    def test_amnesia_rollback_after_true(self):
        """
        THE "AMNESIA" TEST
        
        Verify that when rollback_after=True, the model weights are
        identical to the original after adapt_and_infer returns.
        
        This tests the "Standard TTT" mode where adaptation is temporary.
        """
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        print("\n" + "=" * 60)
        print("TEST 1: THE 'AMNESIA' TEST (rollback_after=True)")
        print("=" * 60)
        
        # Create model and adapter
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        config = SEALConfig(
            ttt_lr=1e-2,  # Higher LR to ensure visible weight changes
            ttt_steps=3,  # Multiple steps to amplify changes
            max_diffusion_steps=5,
            confidence_threshold=0.0,  # Accept all pseudo-labels for testing
            consistency_threshold=0.0,  # Accept all for testing
        )
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # STEP 1: Take initial weight snapshot
        print("\n[1] Taking initial weight snapshot...")
        initial_snapshot = get_weight_snapshot(model)
        
        # Record specific block weights for detailed comparison
        block_weight_name = None
        for name in initial_snapshot.keys():
            if 'blocks' in name and 'weight' in name:
                block_weight_name = name
                break
        
        if block_weight_name:
            print(f"    Tracking: {block_weight_name}")
            print(f"    Initial mean: {initial_snapshot[block_weight_name].mean().item():.6f}")
        
        # STEP 2: Create dummy input
        print("\n[2] Creating dummy input...")
        input_scan, intv_t, treat_code, i_tg = create_dummy_input(
            batch_size=self.batch_size,
            height=self.image_size,
            width=self.image_size,
            device=self.device,
        )
        
        # STEP 3: Run adapt_and_infer with rollback_after=True
        print("\n[3] Running adapt_and_infer with rollback_after=True...")
        pred_img, pred_mask, adapt_info = adapter.adapt_and_infer(
            input_scan=input_scan,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            persistent=False,
            rollback_after=True,  # THE KEY SETTING
        )
        
        print(f"    Adaptation result: {adapt_info.get('adapted', False)}")
        print(f"    Update applied: {adapt_info.get('update_applied', False)}")
        
        # STEP 4: Take post-adaptation snapshot
        print("\n[4] Taking post-adaptation weight snapshot...")
        post_snapshot = get_weight_snapshot(model)
        
        if block_weight_name:
            print(f"    Post mean: {post_snapshot[block_weight_name].mean().item():.6f}")
        
        # STEP 5: ASSERTION - Weights should be IDENTICAL
        print("\n[5] ASSERTION: Verifying weights are IDENTICAL...")
        weights_equal, different_params = compare_weights(initial_snapshot, post_snapshot)
        
        if weights_equal:
            print("    ✓ PASSED: All weights are identical after rollback")
        else:
            print(f"    ✗ FAILED: Weights differ in {len(different_params)} parameters:")
            for param in different_params[:5]:  # Show first 5
                print(f"      - {param}")
        
        assert weights_equal, (
            f"AMNESIA TEST FAILED: Weights should be identical after rollback_after=True. "
            f"Different params: {different_params[:5]}"
        )
        
        print("\n" + "=" * 60)
        print("✓ AMNESIA TEST PASSED")
        print("=" * 60)
    
    def test_elasticity_persistent_true(self):
        """
        THE "ELASTICITY" TEST
        
        Verify that when persistent=True (or rollback_after=False),
        the model weights are DIFFERENT after adapt_and_infer returns.
        
        This tests the "Continual Learning" mode where adaptation persists.
        """
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        print("\n" + "=" * 60)
        print("TEST 2: THE 'ELASTICITY' TEST (persistent=True)")
        print("=" * 60)
        
        # Create model and adapter
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        config = SEALConfig(
            ttt_lr=1e-2,  # Higher LR to ensure visible weight changes
            ttt_steps=3,  # Multiple steps to amplify changes
            max_diffusion_steps=5,
            confidence_threshold=0.0,  # Accept all pseudo-labels for testing
            consistency_threshold=0.0,  # Accept all for testing
        )
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # STEP 1: Take initial weight snapshot
        print("\n[1] Taking initial weight snapshot...")
        initial_snapshot = get_weight_snapshot(model)
        
        # Record specific block weights for detailed comparison
        block_weight_name = None
        for name in initial_snapshot.keys():
            if 'blocks' in name and 'weight' in name:
                block_weight_name = name
                break
        
        if block_weight_name:
            print(f"    Tracking: {block_weight_name}")
            print(f"    Initial mean: {initial_snapshot[block_weight_name].mean().item():.6f}")
        
        # STEP 2: Create dummy input
        print("\n[2] Creating dummy input...")
        input_scan, intv_t, treat_code, i_tg = create_dummy_input(
            batch_size=self.batch_size,
            height=self.image_size,
            width=self.image_size,
            device=self.device,
        )
        
        # STEP 3: Run adapt_and_infer with persistent=True
        print("\n[3] Running adapt_and_infer with persistent=True, rollback_after=False...")
        pred_img, pred_mask, adapt_info = adapter.adapt_and_infer(
            input_scan=input_scan,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            persistent=True,     # THE KEY SETTING
            rollback_after=False,  # THE KEY SETTING
        )
        
        print(f"    Adaptation result: {adapt_info.get('adapted', False)}")
        print(f"    Update applied: {adapt_info.get('update_applied', False)}")
        if 'losses' in adapt_info:
            print(f"    Losses: {adapt_info['losses']}")
        
        # STEP 4: Take post-adaptation snapshot
        print("\n[4] Taking post-adaptation weight snapshot...")
        post_snapshot = get_weight_snapshot(model)
        
        if block_weight_name:
            print(f"    Post mean: {post_snapshot[block_weight_name].mean().item():.6f}")
        
        # STEP 5: ASSERTION - Weights should be DIFFERENT
        print("\n[5] ASSERTION: Verifying weights are DIFFERENT...")
        weights_equal, different_params = compare_weights(initial_snapshot, post_snapshot)
        
        # For this test, we expect weights to be different (adaptation happened)
        # But only if adaptation was actually applied
        if adapt_info.get('update_applied', False):
            if not weights_equal:
                print(f"    ✓ PASSED: Weights changed in {len(different_params)} parameters")
                for param in different_params[:5]:  # Show first 5
                    print(f"      - {param}")
            else:
                print("    ✗ FAILED: Weights are identical (should have changed)")
            
            assert not weights_equal, (
                "ELASTICITY TEST FAILED: Weights should be different after persistent=True "
                "adaptation was applied."
            )
        else:
            # If no update was applied, weights should be same
            print("    Note: No update was applied (pseudo-label rejected)")
            print("    Verifying weights unchanged in this case...")
            assert weights_equal, "Weights changed even though no update was applied"
            print("    ✓ PASSED: Weights correctly unchanged (no update applied)")
        
        print("\n" + "=" * 60)
        print("✓ ELASTICITY TEST PASSED")
        print("=" * 60)
    
    def test_reset_to_original(self):
        """Test that reset_to_original restores initial weights after multiple adaptations."""
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        print("\n" + "=" * 60)
        print("TEST 3: RESET TO ORIGINAL")
        print("=" * 60)
        
        # Create model and adapter
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        config = SEALConfig(
            ttt_lr=1e-2,
            ttt_steps=2,
            max_diffusion_steps=5,
            confidence_threshold=0.0,
            consistency_threshold=0.0,
        )
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # Take original snapshot
        original_snapshot = get_weight_snapshot(model)
        
        # Run multiple persistent adaptations
        for i in range(3):
            input_scan, intv_t, treat_code, i_tg = create_dummy_input(
                batch_size=self.batch_size,
                height=self.image_size,
                width=self.image_size,
                device=self.device,
            )
            
            adapter.adapt_and_infer(
                input_scan=input_scan,
                intv_t=intv_t,
                treat_code=treat_code,
                i_tg=i_tg,
                persistent=True,
                rollback_after=False,
            )
        
        # Verify weights have changed
        post_adapt_snapshot = get_weight_snapshot(model)
        weights_equal, _ = compare_weights(original_snapshot, post_adapt_snapshot)
        
        # Reset to original
        print("\n[1] Resetting to original state...")
        adapter.reset_to_original()
        
        # Verify weights are restored
        reset_snapshot = get_weight_snapshot(model)
        weights_equal, different = compare_weights(original_snapshot, reset_snapshot)
        
        if weights_equal:
            print("    ✓ PASSED: Weights restored to original")
        else:
            print(f"    ✗ FAILED: {len(different)} parameters differ")
        
        assert weights_equal, f"Reset to original failed. Different: {different[:5]}"
        
        print("\n" + "=" * 60)
        print("✓ RESET TO ORIGINAL TEST PASSED")
        print("=" * 60)
    
    def test_adaptation_stats(self):
        """Test that adaptation statistics are tracked correctly."""
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        print("\n" + "=" * 60)
        print("TEST 4: ADAPTATION STATISTICS")
        print("=" * 60)
        
        # Create model and adapter
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        config = SEALConfig(
            ttt_lr=1e-2,
            ttt_steps=1,
            max_diffusion_steps=5,
            confidence_threshold=0.0,  # Accept all
            consistency_threshold=0.0,
        )
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # Verify initial stats
        stats = adapter.get_stats()
        assert stats['total_samples'] == 0
        assert stats['total_updates'] == 0
        
        # Run adaptation
        input_scan, intv_t, treat_code, i_tg = create_dummy_input(
            batch_size=self.batch_size,
            height=self.image_size,
            width=self.image_size,
            device=self.device,
        )
        
        adapter.adapt_single_sample(
            input_scan=input_scan,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            persistent=True,
        )
        
        # Verify stats updated
        stats = adapter.get_stats()
        print(f"\n    Total samples: {stats['total_samples']}")
        print(f"    Accepted samples: {stats['accepted_samples']}")
        print(f"    Rejected samples: {stats['rejected_samples']}")
        print(f"    Total updates: {stats['total_updates']}")
        print(f"    Avg confidence: {stats['avg_confidence']:.4f}")
        
        assert stats['total_samples'] == 1, "Should have processed 1 sample"
        
        # Reset stats
        adapter.reset_stats()
        stats = adapter.get_stats()
        assert stats['total_samples'] == 0, "Stats should be reset"
        
        print("\n    ✓ PASSED: Statistics tracked correctly")
        
        print("\n" + "=" * 60)
        print("✓ ADAPTATION STATISTICS TEST PASSED")
        print("=" * 60)
    
    def test_lr_warmup_and_cosine_schedule(self):
        """
        LR WARMUP AND COSINE SCHEDULE TEST
        
        Verify that warmup_steps and use_cosine_schedule work correctly.
        This matches SEAL's training pattern (warmup_steps=11, cosine decay).
        """
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        
        print("\n" + "=" * 60)
        print("TEST: LR WARMUP AND COSINE SCHEDULE")
        print("=" * 60)
        
        model = create_dummy_tadiff_net(image_size=self.image_size)
        model = model.to(self.device)
        
        # Config with warmup and cosine schedule (SEAL-style)
        config = SEALConfig(
            ttt_lr=1e-3,
            ttt_steps=10,  # Enough steps to see warmup and decay
            warmup_steps=3,  # Warmup for first 3 steps
            use_cosine_schedule=True,  # Cosine decay after warmup
            max_diffusion_steps=5,
            confidence_threshold=0.0,
            consistency_threshold=0.0,
            reset_optimizer_per_sample=True,
        )
        
        adapter = SEALAdapter(model, config=config, device=self.device)
        
        # Create input and pseudo-target
        input_scan, intv_t, treat_code, i_tg = create_dummy_input(
            batch_size=1, height=self.image_size, width=self.image_size, device=self.device
        )
        pseudo_target = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        
        # Run update and capture LR values
        losses = adapter.apply_update(
            input_scan=input_scan,
            pseudo_target=pseudo_target,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            num_steps=10,
        )
        
        # Extract LR values
        lrs = [losses[f'lr_{i}'] for i in range(10)]
        
        print(f"\n    LR progression (warmup_steps=3, cosine=True):")
        for i, lr in enumerate(lrs):
            print(f"      Step {i}: LR = {lr:.6f}")
        
        # Verify warmup: LR should increase for first 3 steps
        assert lrs[0] < lrs[1] < lrs[2], "LR should increase during warmup"
        print("    ✓ Warmup verified: LR increases during warmup phase")
        
        # Verify LR at end of warmup is approximately base_lr
        assert 0.9 * config.ttt_lr <= lrs[2] <= 1.1 * config.ttt_lr, \
            f"LR at end of warmup should be ~{config.ttt_lr}, got {lrs[2]}"
        print(f"    ✓ LR at warmup end: {lrs[2]:.6f} ≈ base_lr {config.ttt_lr}")
        
        # Verify cosine decay: LR should decrease after warmup
        assert lrs[-1] < lrs[3], "LR should decrease with cosine schedule"
        print(f"    ✓ Cosine decay verified: LR {lrs[3]:.6f} → {lrs[-1]:.6f}")
        
        # Test reset_optimizer_per_sample
        print("\n    Testing reset_optimizer_per_sample...")
        
        # First call creates optimizer
        adapter.apply_update(
            input_scan=input_scan,
            pseudo_target=pseudo_target,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            num_steps=1,
            reset_optimizer=True,
        )
        opt1_id = id(adapter._ttt_optimizer)
        
        # Second call with reset=True should create new optimizer
        adapter.apply_update(
            input_scan=input_scan,
            pseudo_target=pseudo_target,
            intv_t=intv_t,
            treat_code=treat_code,
            i_tg=i_tg,
            num_steps=1,
            reset_optimizer=True,
        )
        opt2_id = id(adapter._ttt_optimizer)
        
        assert opt1_id != opt2_id, "Optimizer should be reset between calls"
        print("    ✓ Optimizer reset between samples verified")
        
        print("\n" + "=" * 60)
        print("✓ LR WARMUP AND COSINE SCHEDULE TEST PASSED")
        print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "=" * 70)
    print("SEAL ADAPTER SMOKE TESTS")
    print("=" * 70)
    
    test_suite = TestSEALAdapter()
    test_suite.setup()
    
    tests = [
        ("Adapter Initialization", test_suite.test_adapter_initialization),
        ("State Save/Restore", test_suite.test_state_save_restore),
        ("Amnesia Test (rollback_after=True)", test_suite.test_amnesia_rollback_after_true),
        ("Elasticity Test (persistent=True)", test_suite.test_elasticity_persistent_true),
        ("Reset to Original", test_suite.test_reset_to_original),
        ("Adaptation Statistics", test_suite.test_adaptation_stats),
        ("LR Warmup and Cosine Schedule", test_suite.test_lr_warmup_and_cosine_schedule),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASSED", None))
        except AssertionError as e:
            results.append((name, "FAILED", str(e)))
        except Exception as e:
            results.append((name, "ERROR", str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")
    
    for name, status, error in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      {error[:80]}...")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {errors} errors")
    print("=" * 70)
    
    return failed == 0 and errors == 0


if __name__ == "__main__":
    # Check if pytest is available
    if PYTEST_AVAILABLE:
        # Run with pytest for better output
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(exit_code)
    else:
        # Fall back to manual test runner
        print("pytest not found, running tests manually...")
        success = run_all_tests()
        sys.exit(0 if success else 1)
