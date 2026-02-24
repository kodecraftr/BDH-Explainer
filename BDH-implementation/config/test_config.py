"""
Test Configuration for TaDiff Model.

Compatible with TaDiff-Net configuration.
Reference: https://github.com/samleoqh/TaDiff-Net/blob/main/config/test_config.py
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TestConfig:
    """Configuration for testing TaDiff model.
    
    Attributes:
        data_root: Root directory for data files
        save_path: Directory to save test results
        model_checkpoint: Path to model checkpoint
        patient_ids: List of patient IDs to test
        model_channels: Base model channels
        num_heads: Number of attention heads
        num_res_blocks: Number of residual blocks
        diffusion_steps: Number of diffusion sampling steps
        target_session_idx: Target session index for prediction
        num_samples: Number of prediction samples to generate
        min_tumor_size: Minimum tumor volume threshold
        top_k_slices: Number of top tumor slices to process
        npz_keys: Data keys for loading
        colors: Color mapping for visualization
        mask_threshold: Threshold for mask binarization
        dice_thresholds: Thresholds for Dice evaluation
    """
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    data_root: str = "./data/sailor_npy"
    save_path: str = "./test_results"
    model_checkpoint: str = "./ckpt/model.ckpt"
    
    # Patient IDs to test
    patient_ids: List[str] = field(default_factory=lambda: ['sub-17'])
    
    # =========================================================================
    # Model Architecture
    # =========================================================================
    # For TaDiff-Net UNet-based model
    model_channels: int = 256
    num_heads: int = 16
    num_res_blocks: int = 1
    bdh_expansion: int = 4
    
    # For TaDiff-DiT model (our extended version)
    hidden_size: int = 1024
    depth: int = 18
    patch_size: int = 8
    
    # Common parameters
    image_size: int = 192
    in_channels: int = 13  # 4 sessions * 3 modalities + target session
    out_channels: int = 7  # 4 sessions * 1 mask + 3 image channels
    
    # =========================================================================
    # Diffusion Parameters
    # =========================================================================
    max_T: int = 2000
    diffusion_steps: int = 1000  # Sampling steps (<= max_T)
    ddpm_schedule: str = 'cosine'  # 'linear', 'cosine'
    
    # =========================================================================
    # Test Parameters
    # =========================================================================
    target_session_idx: int = 3  # Predict session 4 (0-indexed)
    num_samples: int = 5  # Number of prediction samples for ensemble
    min_tumor_size: int = 20  # Minimum tumor volume (voxels)
    top_k_slices: int = 3  # Number of top slices to process
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    npz_keys: List[str] = field(default_factory=lambda: ['image', 'label', 'days', 'treatment'])
    num_modalities: int = 4  # T1, T1c, FLAIR, T2
    prefix: str = ''  # Filename prefix (e.g., 'sub-')
    
    # =========================================================================
    # Visualization Settings
    # =========================================================================
    colors: dict = field(default_factory=lambda: {
        0: (0, 0, 0),      # background, black
        1: (0, 255, 0),    # class 1, green (edema/growth)
        2: (0, 0, 255),    # class 2, blue (shrinkage)
        3: (255, 0, 0),    # class 3, red (enhancing/stable)
    })
    
    # =========================================================================
    # Evaluation Thresholds
    # =========================================================================
    mask_threshold: float = 0.49
    dice_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    
    # =========================================================================
    # Runtime Settings
    # =========================================================================
    device: str = 'auto'  # 'auto', 'cuda', 'cpu', 'mps'
    num_workers: int = 0
    verbose: bool = True
    save_visualizations: bool = True
    
    def get_device(self) -> str:
        """Get the appropriate device string."""
        import torch
        if self.device != 'auto':
            return self.device
        
        if torch.cuda.is_available():
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'


@dataclass
class TrainConfig:
    """Configuration for training TaDiff model.
    
    Extends TestConfig with training-specific parameters.
    """
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    data_root: str = "./data/sailor_npy"
    save_path: str = "./training_outputs"
    
    # =========================================================================
    # Model Architecture
    # =========================================================================
    image_size: int = 192
    in_channels: int = 12
    out_channels: int = 7
    model_channels: int = 128
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 16
    
    # =========================================================================
    # Diffusion Parameters
    # =========================================================================
    max_T: int = 1000
    ddpm_schedule: str = 'linear'
    
    # =========================================================================
    # Optimizer and Learning Rate
    # =========================================================================
    optimizer: str = 'adamw'
    lr: float = 5e-4
    weight_decay: float = 3e-5
    warmup_steps: int = 100
    lr_gamma: float = 0.585
    lrdecay_cosine: bool = True
    
    # =========================================================================
    # Training Parameters
    # =========================================================================
    max_epochs: int = 3000
    max_steps: int = 60000
    batch_size: int = 1
    sw_batch: int = 16  # Sliding window batch size
    accumulate_grad_batches: int = 4
    
    # =========================================================================
    # Loss Configuration
    # =========================================================================
    loss_type: str = 'ssim'  # 'ssim', 'mse', 'combined'
    loss_weights: dict = field(default_factory=lambda: {
        'diffusion': 1.0,
        'segmentation': 0.1,
        'ssim': 0.1,
    })
    
    # =========================================================================
    # Validation and Checkpointing
    # =========================================================================
    val_interval_epoch: int = 10
    ckpt_save_top_k: int = 3
    ckpt_save_last: bool = True
    ckpt_monitor: str = 'val_loss'
    ckpt_mode: str = 'min'
    resume_from_ckpt: Optional[str] = None
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    num_workers: int = 4
    train_split: float = 0.8
    augment: bool = True
    
    # =========================================================================
    # Hardware
    # =========================================================================
    precision: int = 32  # 16 or 32
    device: str = 'auto'
    
    def get_device(self) -> str:
        """Get the appropriate device string."""
        import torch
        if self.device != 'auto':
            return self.device
        
        if torch.cuda.is_available():
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'


@dataclass  
class InferenceConfig:
    """Configuration for inference without ground truth."""
    
    # Data paths
    data_root: str = "./data/sailor_npy"
    output_dir: str = "./inference_results"
    model_checkpoint: str = "./ckpt/model.ckpt"
    
    # Patient selection
    patient_ids: List[str] = field(default_factory=lambda: ['sub-17'])
    prefix: str = ''
    
    # Model parameters
    image_size: int = 192
    model_channels: int = 32
    num_heads: int = 1
    num_res_blocks: int = 1
    
    # Inference parameters
    diffusion_steps: int = 50  # Fewer steps for faster inference
    num_samples: int = 4
    slice_idx: Optional[int] = None  # None = use middle slice
    
    # Target prediction
    input_day: int = 10  # Days after last session
    input_treatment: int = 1  # Treatment code (0: CRT, 1: TMZ)
    
    # SEAL TTT parameters
    enable_ttt: bool = False
    ttt_lr: float = 1e-4
    ttt_steps: int = 5
    
    # Runtime
    device: str = 'auto'
    save_visualizations: bool = True
    
    def get_device(self) -> str:
        """Get the appropriate device string."""
        import torch
        if self.device != 'auto':
            return self.device
        
        if torch.cuda.is_available():
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
