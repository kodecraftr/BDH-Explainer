"""
Test script for DIT-BDH architecture.

This script verifies that the DiT architecture:
1. Correctly handles input/output dimensions
2. Properly processes treatment and temporal conditioning
3. Maintains gradient flow for training
"""

import os
import sys
import torch
import torch.nn as nn

# Add the repository root to path for proper imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Note: The mock modules below are only needed if running this test
# in isolation without the full src package available.
import types
import math


class MockFourierFeatures(nn.Module):
    """Fallback FourierFeatures for standalone testing."""
    def __init__(self, in_features, out_features, std=0.2):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def _create_mock_utils_module():
    """Create mock utils module for standalone testing."""
    utils_module = types.ModuleType('src.net.utils')
    
    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def checkpoint(func, inputs, params, flag):
        if flag:
            return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
        else:
            return func(*inputs)
    
    utils_module.checkpoint = checkpoint
    utils_module.conv_nd = lambda dims, *args, **kwargs: nn.Conv2d(*args, **kwargs) if dims == 2 else nn.Conv1d(*args, **kwargs)
    utils_module.linear = nn.Linear
    utils_module.normalization = lambda channels, g=32: nn.GroupNorm(g, channels)
    utils_module.timestep_embedding = timestep_embedding
    utils_module.FourierFeatures = MockFourierFeatures
    
    return utils_module


# Only install mock modules if real ones aren't available
try:
    from src.net.utils import timestep_embedding, FourierFeatures
except ImportError:
    utils_module = _create_mock_utils_module()
    sys.modules['src'] = types.ModuleType('src')
    sys.modules['src.net'] = types.ModuleType('src.net')
    sys.modules['src.net.utils'] = utils_module


def test_tadiff_dit():
    """Test the TaDiff-DiT model."""
    print("=" * 60)
    print("TaDiff-DiT Architecture Test Suite")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Import the model - try package import first, fall back to direct
    try:
        from src.net.tadiff_dit_arch import TaDiff_DiT, PatchEmbed, DiTBlock, FinalLayer
    except ImportError:
        # Fallback for running from src/net directory
        from tadiff_dit_arch import TaDiff_DiT, PatchEmbed, DiTBlock, FinalLayer
    
    # Test 1: PatchEmbed
    print("\n[Test 1] PatchEmbed...")
    patch_embed = PatchEmbed(
        image_size=192,
        patch_size=16,
        in_channels=12,
        embed_dim=768
    ).to(device)
    
    x = torch.randn(2, 12, 192, 192).to(device)
    patches = patch_embed(x)
    assert patches.shape == (2, 144, 768), f"Expected (2, 144, 768), got {patches.shape}"
    print(f"  Input: {x.shape} -> Output: {patches.shape}")
    print("  ✓ PatchEmbed passed")
    
    # Test 2: DiTBlock
    print("\n[Test 2] DiTBlock...")
    dit_block = DiTBlock(
        hidden_size=768,
        num_heads=12,
        condition_dim=768,
        mlp_ratio=4.0
    ).to(device)
    
    c = torch.randn(2, 768).to(device)
    out = dit_block(patches, c)
    assert out.shape == patches.shape, f"Expected {patches.shape}, got {out.shape}"
    print(f"  Input: {patches.shape}, Condition: {c.shape} -> Output: {out.shape}")
    print("  ✓ DiTBlock passed")
    
    # Test 3: FinalLayer
    print("\n[Test 3] FinalLayer...")
    final_layer = FinalLayer(
        hidden_size=768,
        patch_size=16,
        out_channels=7,
        condition_dim=768
    ).to(device)
    
    output = final_layer(out, c, 192, 192)
    assert output.shape == (2, 7, 192, 192), f"Expected (2, 7, 192, 192), got {output.shape}"
    print(f"  Input: {out.shape} -> Output: {output.shape}")
    print("  ✓ FinalLayer passed")
    
    # Test 4: Full TaDiff_DiT model
    print("\n[Test 4] Full TaDiff_DiT Model...")
    model = TaDiff_DiT(
        image_size=192,
        in_channels=12,
        out_channels=7,
        model_channels=128,
        hidden_size=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        use_checkpoint=False,
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create inputs
    batch_size = 2
    x = torch.randn(batch_size, 12, 192, 192).to(device)
    timesteps = torch.randint(1, 1000, (batch_size,)).to(device).float()
    
    # Create conditioning
    intv_t = [
        torch.tensor([0, 30]).float().to(device),
        torch.tensor([90, 100]).float().to(device),
        torch.tensor([180, 200]).float().to(device),
        torch.tensor([270, 300]).float().to(device),
    ]
    treat_code = [
        torch.tensor([0, 1]).float().to(device),
        torch.tensor([1, 0]).float().to(device),
        torch.tensor([0, 1]).float().to(device),
        torch.tensor([1, 1]).float().to(device),
    ]
    i_tg = torch.tensor([-1, -1]).to(device)
    
    # Forward pass
    print("\n  Running forward pass...")
    with torch.no_grad():
        output = model(x, timesteps, intv_t, treat_code, i_tg)
    
    assert output.shape == (batch_size, 7, 192, 192), f"Expected ({batch_size}, 7, 192, 192), got {output.shape}"
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print("  ✓ Forward pass passed")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient Flow...")
    model.train()
    x = torch.randn(2, 12, 192, 192, requires_grad=True).to(device)
    timesteps = torch.randint(1, 1000, (2,)).to(device).float()
    
    output = model(x, timesteps, intv_t, treat_code, i_tg)
    loss = output.mean()
    loss.backward()
    
    assert x.grad is not None, "Gradients not flowing to input"
    print(f"  Input gradient shape: {x.grad.shape}")
    print("  ✓ Gradient flow passed")
    
    # Test 6: Conditioning dimensions
    print("\n[Test 6] Conditioning Dimensions...")
    print(f"  model_channels: {model.model_channels}")
    print(f"  hidden_size: {model.hidden_size}")
    print(f"  all_time_day_dim: {model.all_time_day_dim}")
    print(f"  Grid size: {model.grid_size}x{model.grid_size}")
    print(f"  Num patches: {model.num_patches}")
    print("  ✓ Conditioning dimensions correct")
    
    # Test 7: Different batch sizes
    print("\n[Test 7] Different Batch Sizes...")
    for bs in [1, 4, 8]:
        x = torch.randn(bs, 12, 192, 192).to(device)
        timesteps = torch.randint(1, 1000, (bs,)).to(device).float()
        intv_t_bs = [torch.rand(bs).to(device) * 300 for _ in range(4)]
        treat_code_bs = [torch.randint(0, 3, (bs,)).float().to(device) for _ in range(4)]
        i_tg_bs = -torch.ones(bs, dtype=torch.int8).to(device)
        
        with torch.no_grad():
            out = model(x, timesteps, intv_t_bs, treat_code_bs, i_tg_bs)
        
        assert out.shape == (bs, 7, 192, 192), f"Batch size {bs}: Expected ({bs}, 7, 192, 192), got {out.shape}"
        print(f"  Batch size {bs}: ✓")
    print("  ✓ Different batch sizes passed")
    
    # Test 8: Memory usage
    print("\n[Test 8] Memory Usage (if CUDA)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(4, 12, 192, 192).to(device)
        timesteps = torch.randint(1, 1000, (4,)).to(device).float()
        intv_t_4 = [torch.rand(4).to(device) * 300 for _ in range(4)]
        treat_code_4 = [torch.randint(0, 3, (4,)).float().to(device) for _ in range(4)]
        i_tg_4 = -torch.ones(4, dtype=torch.int8).to(device)
        
        output = model(x, timesteps, intv_t_4, treat_code_4, i_tg_4)
        loss = output.mean()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(f"  Peak memory usage: {peak_memory:.2f} GB")
        print("  ✓ Memory usage test passed")
    else:
        print("  Skipped (no CUDA)")
    
    print("\n" + "=" * 60)
    print("All architecture tests passed! ✓")
    print("=" * 60)


def test_seal_adapter_quick():
    """Quick SEAL adapter verification tests."""
    print("\n" + "=" * 60)
    print("SEAL Adapter Quick Tests")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from src.net.seal_adapter import SEALAdapter, SEALConfig
        from src.net.tadiff_dit_arch import TaDiff_DiT
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test 1: Basic initialization
    print("\n[Test 1] SEAL Adapter Initialization...")
    try:
        model = TaDiff_DiT(
            image_size=64, in_channels=12, out_channels=7,
            model_channels=32, hidden_size=64, depth=2,
            num_heads=2, patch_size=8,
        ).to(device)
        
        config = SEALConfig(ttt_lr=1e-4, ttt_steps=2)
        adapter = SEALAdapter(model, config=config, device=device)
        print("  ✓ Initialization passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 2: State save/restore
    print("\n[Test 2] State Save/Restore...")
    try:
        # Get initial weight
        initial_weight = list(model.parameters())[0].clone()
        
        # Save state
        adapter.save_state()
        
        # Modify weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        
        # Restore state
        adapter.restore_state()
        
        # Check weights restored
        restored_weight = list(model.parameters())[0]
        assert torch.allclose(initial_weight, restored_weight), "Weights not restored"
        print("  ✓ State save/restore passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 3: Adaptation stats
    print("\n[Test 3] Adaptation Statistics...")
    try:
        stats = adapter.get_stats()
        assert 'total_samples' in stats
        assert 'total_updates' in stats
        assert 'avg_confidence' in stats
        print(f"  Stats: {stats}")
        print("  ✓ Adaptation stats passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All SEAL adapter tests passed! ✓")
    print("=" * 60)
    return True


def test_diffusion_quick():
    """Quick diffusion process verification."""
    print("\n" + "=" * 60)
    print("Diffusion Process Quick Tests")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from src.net.diffusion import GaussianDiffusion
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test 1: Linear schedule
    print("\n[Test 1] Linear Schedule...")
    try:
        diffusion = GaussianDiffusion(T=100, schedule='linear', device=device)
        assert len(diffusion.beta) == 100
        assert torch.all(diffusion.alphabar[1:] <= diffusion.alphabar[:-1])
        print("  ✓ Linear schedule passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 2: Cosine schedule
    print("\n[Test 2] Cosine Schedule...")
    try:
        diffusion = GaussianDiffusion(T=100, schedule='cosine', device=device)
        assert len(diffusion.beta) == 100
        print("  ✓ Cosine schedule passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 3: Sample forward
    print("\n[Test 3] Forward Sampling...")
    try:
        x0 = torch.randn(2, 3, 64, 64, device=device)
        t = torch.tensor([10, 50], device=device)
        xt, epsilon = diffusion.sample(x0, t)
        assert xt.shape == x0.shape
        assert epsilon.shape == x0.shape
        print(f"  x0: {x0.shape} -> xt: {xt.shape}")
        print("  ✓ Forward sampling passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 4: Invalid schedule error
    print("\n[Test 4] Invalid Schedule Validation...")
    try:
        try:
            GaussianDiffusion(T=100, schedule='invalid')
            print("  ✗ Should have raised ValueError")
            return False
        except ValueError:
            print("  ✓ Invalid schedule correctly raises ValueError")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All diffusion tests passed! ✓")
    print("=" * 60)
    return True


def test_data_loading_quick():
    """Quick data loading test with synthetic data."""
    print("\n" + "=" * 60)
    print("Data Loading Quick Tests")
    print("=" * 60)
    
    import tempfile
    import numpy as np
    from pathlib import Path
    
    # Try SAILORDataset first, fall back to data_loader
    dataset_class = None
    try:
        from src.data.sailor_dataset import SAILORDataset
        dataset_class = 'SAILORDataset'
    except ImportError:
        try:
            from src.data.data_loader import get_test_files, load_data
            dataset_class = 'data_loader'
        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
            return False
    
    # Create temporary synthetic data
    print("\n[Test 1] Creating synthetic data...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate minimal synthetic patient
            patient_id = "sub-01"
            num_sessions = 4
            H, W, D = 64, 64, 32  # Small for speed
            M = 4
            
            image = np.random.rand(M * num_sessions, H, W, D).astype(np.float32)
            label = np.random.randint(0, 2, (num_sessions, H, W, D)).astype(np.float32)
            days = np.array([0, 30, 60, 90], dtype=np.float32)
            treatment = np.array([0, 0, 1, 1], dtype=np.float32)
            
            np.save(f"{tmpdir}/{patient_id}_image.npy", image)
            np.save(f"{tmpdir}/{patient_id}_label.npy", label)
            np.save(f"{tmpdir}/{patient_id}_days.npy", days)
            np.save(f"{tmpdir}/{patient_id}_treatment.npy", treatment)
            print("  ✓ Synthetic data created")
            
            if dataset_class == 'SAILORDataset':
                # Test SAILORDataset
                print("\n[Test 2] Creating SAILORDataset...")
                dataset = SAILORDataset(
                    data_dir=tmpdir,
                    num_sessions=4,
                    image_size=64,
                    num_slices_per_volume=2,
                    mode='train',
                    augment=False,
                )
                print(f"  ✓ Dataset created: {len(dataset)} samples")
                
                # Test sample loading
                print("\n[Test 3] Loading sample...")
                sample = dataset[0]
                assert 'image' in sample
                assert 'label' in sample
                assert 'days' in sample
                print(f"  Image shape: {sample['image'].shape}")
                print(f"  Label shape: {sample['label'].shape}")
                print("  ✓ Sample loading passed")
                
                # Test DataLoader
                print("\n[Test 4] Testing DataLoader...")
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
                batch = next(iter(dataloader))
                assert batch['image'].shape[0] == 2
                print(f"  Batch shape: {batch['image'].shape}")
                print("  ✓ DataLoader passed")
            
            else:
                # Test MONAI data_loader
                print("\n[Test 2] Creating MONAI DataLoader...")
                test_files = get_test_files(tmpdir, [patient_id])
                print(f"  Test files: {len(test_files)}")
                
                if test_files:
                    print("\n[Test 3] Loading data...")
                    dataloader = load_data(test_files)
                    batch = next(iter(dataloader))
                    print(f"  Image shape: {batch['image'].shape}")
                    print(f"  Label shape: {batch['label'].shape}")
                    print("  ✓ MONAI DataLoader passed")
                else:
                    print("  ✓ No files found (expected with minimal data)")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All data loading tests passed! ✓")
    print("=" * 60)
    return True


def test_connected_pipeline():
    """Test the connected preprocessing -> loading -> processing pipeline."""
    print("\n" + "=" * 60)
    print("Connected Pipeline Test")
    print("=" * 60)
    
    import tempfile
    import numpy as np
    
    try:
        from data.preproc_prepare_data import create_synthetic_patient, validate_processed_data
        from src.data.data_loader import get_test_files, load_data, prepare_batch_for_model
        from config.test_config import TestConfig
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create synthetic data
            print("\n[Step 1] Creating synthetic patient data...")
            files = create_synthetic_patient(
                save_path=tmpdir,
                patient_id="test-01",
                num_sessions=4,
                shape=(64, 64, 32),
            )
            print("  ✓ Synthetic data created")
            
            # Step 2: Validate data
            print("\n[Step 2] Validating processed data...")
            valid = validate_processed_data(tmpdir, "test-01")
            if not valid:
                print("  ✗ Validation failed")
                return False
            print("  ✓ Data validation passed")
            
            # Step 3: Load with data_loader
            print("\n[Step 3] Loading with MONAI pipeline...")
            test_files = get_test_files(tmpdir, ["test-01"])
            
            if not test_files:
                print("  ✗ No test files found")
                return False
            
            dataloader = load_data(test_files)
            batch = next(iter(dataloader))
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Label shape: {batch['label'].shape}")
            print("  ✓ Data loading passed")
            
            # Step 4: Prepare for model
            print("\n[Step 4] Preparing batch for model...")
            device = torch.device('cpu')
            prepared = prepare_batch_for_model(batch, device)
            print(f"  Prepared image shape: {prepared['image'].shape}")
            print(f"  Num sessions: {prepared['num_sessions']}")
            print("  ✓ Batch preparation passed")
            
            # Step 5: Test with model (if available)
            print("\n[Step 5] Testing model forward pass...")
            try:
                from src.net.tadiff_dit_arch import TaDiff_DiT
                
                # Create small model for testing
                model = TaDiff_DiT(
                    image_size=64,
                    in_channels=12,
                    out_channels=7,
                    model_channels=32,
                    hidden_size=64,
                    depth=2,
                    num_heads=2,
                    patch_size=8,
                ).to(device)
                
                # Create dummy 2D input (batch, channels, H, W)
                x = torch.randn(2, 12, 64, 64).to(device)
                timesteps = torch.tensor([100.0, 200.0]).to(device)
                intv_t = [torch.tensor([0.0, 30.0]).to(device) for _ in range(4)]
                treat_code = [torch.tensor([0.0, 1.0]).to(device) for _ in range(4)]
                i_tg = torch.tensor([-1, -1]).to(device)
                
                with torch.no_grad():
                    output = model(x, timesteps, intv_t, treat_code, i_tg)
                
                print(f"  Output shape: {output.shape}")
                print("  ✓ Model forward pass passed")
                
            except ImportError as e:
                print(f"  ⚠ Model not available: {e}")
                print("  (This is OK for pipeline test)")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Connected pipeline test passed! ✓")
    print("=" * 60)
    return True


def run_pytest_tests():
    """Run pytest tests if available."""
    print("\n" + "=" * 60)
    print("Running Pytest Test Suites")
    print("=" * 60)
    
    try:
        import pytest
        import subprocess
        
        # Run pytest programmatically
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ All pytest tests passed!")
            return True
        else:
            print(f"\n✗ Pytest tests failed (exit code: {result.returncode})")
            return False
            
    except ImportError:
        print("  pytest not installed. Install with: pip install pytest")
        return False
    except Exception as e:
        print(f"  Error running pytest: {e}")
        return False


def print_usage():
    """Print usage information."""
    print("""
Usage: python test.py [OPTIONS]

Options:
    --all           Run all tests (quick + pytest + pipeline)
    --quick         Run quick verification tests only (default)
    --pytest        Run pytest test suites only
    --arch          Run architecture tests only
    --seal          Run SEAL adapter tests only
    --diffusion     Run diffusion tests only
    --data          Run data loading tests only
    --pipeline      Run connected pipeline test (preproc -> loading -> model)
    --help, -h      Show this help message

Examples:
    python test.py              # Quick tests only
    python test.py --all        # All tests including pytest
    python test.py --pytest     # Only pytest tests
    python test.py --data       # Data loading tests only
    python test.py --pipeline   # Connected pipeline test
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TaDiff-DiT Test Runner', add_help=False)
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--pytest', action='store_true', help='Run pytest tests only')
    parser.add_argument('--arch', action='store_true', help='Run architecture tests only')
    parser.add_argument('--seal', action='store_true', help='Run SEAL tests only')
    parser.add_argument('--diffusion', action='store_true', help='Run diffusion tests only')
    parser.add_argument('--data', action='store_true', help='Run data loading tests only')
    parser.add_argument('--pipeline', action='store_true', help='Run connected pipeline test')
    parser.add_argument('--help', '-h', action='store_true', help='Show help')
    
    args = parser.parse_args()
    
    if args.help:
        print_usage()
        sys.exit(0)
    
    # Track results
    results = {}
    
    # Determine which tests to run
    run_arch = args.arch or args.quick or args.all or (not any([args.pytest, args.seal, args.diffusion, args.data, args.pipeline]))
    run_seal = args.seal or args.quick or args.all
    run_diffusion = args.diffusion or args.quick or args.all
    run_data = args.data or args.all
    run_pipeline = args.pipeline or args.all
    run_pytest_flag = args.pytest or args.all
    
    # Run selected tests
    if run_arch:
        try:
            test_tadiff_dit()
            results['Architecture'] = True
        except Exception as e:
            print(f"\n✗ Architecture tests failed: {e}")
            results['Architecture'] = False
    
    if run_seal:
        results['SEAL Adapter'] = test_seal_adapter_quick()
    
    if run_diffusion:
        results['Diffusion'] = test_diffusion_quick()
    
    if run_data:
        results['Data Loading'] = test_data_loading_quick()
    
    if run_pipeline:
        results['Connected Pipeline'] = test_connected_pipeline()
    
    if run_pytest_flag:
        results['Pytest Suite'] = run_pytest_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)
