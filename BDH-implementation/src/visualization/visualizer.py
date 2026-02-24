

"""
Visualization utilities for TaDiff model results.
Handles plotting, image saving, and visualization of uncertainty maps.
"""
import os
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image, ImageFilter

class Visualizer:
    def __init__(self, colors: dict, bg_color='black', fg_color='white'):
        self.colors = colors
        self.bg_color = bg_color
        self.fg_color = fg_color
        
        # Set matplotlib parameters
        mpl.rc('image', cmap='gray')
        plt.rcParams["text.color"] = fg_color
        plt.rcParams["axes.labelcolor"] = fg_color
        plt.rcParams["xtick.color"] = fg_color
        plt.rcParams["ytick.color"] = fg_color
    
    def to_pil(self, arr, to_rgb=False):
        """Convert numpy array to PIL Image"""
        normalized = ((arr - arr.min()) / (arr.max() - arr.min() + 1.e-8)) * 255.9
        if to_rgb:
            return Image.fromarray(normalized.astype(np.uint8)).convert('RGB')
        return Image.fromarray(normalized.astype(np.uint8))
    
    def plot_uncertainty(self, arr, save_path, overlay=None):
        """Plot uncertainty map with optional overlay"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        cx = np.arange(arr.shape[1])
        cy = np.arange(arr.shape[0])
        X, Y = np.meshgrid(cx, cy)
        
        ax.set_aspect('equal')
        col1 = ax.pcolormesh(X, Y, arr, cmap='magma', 
                            vmin=arr.min(), vmax=arr.max())
        
        if overlay is not None:
            ax.pcolormesh(X, Y, overlay, cmap='gray', alpha=0.35)
        
        ax.axis([cx.min(), cx.max(), cy.max(), cy.min()])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.06)
        cbar = fig.colorbar(col1, cax=cax, extend='max')
        cbar.ax.tick_params(labelsize=18)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor=self.bg_color)
        plt.close()
    
    def draw_contour(self, img, mask, rgb=(255, 0, 0)):
        """Draw contour of mask on image"""
        # Create binary mask
        binary_mask = mask.point(lambda p: p >= 128 and 255)
        
        # Find edges
        edges = binary_mask.filter(ImageFilter.FIND_EDGES)
        edges_np = np.array(edges)
        
        # Convert image to RGBA if not already
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert('RGBA')
        img_np = np.array(img)
        
        # Draw edges in specified color
        img_np[np.nonzero(edges_np)] = [*rgb, 255]
        
        return Image.fromarray(img_np)
    
    def overlay_maps(self, img, curr_mask, past_mask, transparency=0.4):
        """Overlay current and past masks on image"""
        img = img.convert('RGBA')
        combined_mask = self.to_binary(np.array(curr_mask)) + 2 * self.to_binary(np.array(past_mask))
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay = np.array(overlay)
        
        for k, v in self.colors.items():
            alpha = 0 if (k == 0 or k == 3) else transparency
            overlay[combined_mask == k] = [*v, int(alpha * 255)]
        
        overlay = Image.fromarray(np.uint8(overlay))
        return Image.alpha_composite(img, overlay)
    
    @staticmethod
    def to_binary(arr, threshold=0.5):
        """Convert array to binary mask"""
        th = arr.max() * threshold
        return (arr >= th).astype(np.uint8)

    def save_validation_preview(
        self,
        pred_img: np.ndarray,
        gt_img: np.ndarray,
        pred_mask: Optional[np.ndarray],
        save_path: str,
        title: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Save a compact validation preview figure.

        Args:
            pred_img: Predicted image [3, H, W] in [-1, 1] or [0, 1]
            gt_img: Ground-truth image [3, H, W] in [-1, 1] or [0, 1]
            pred_mask: Optional predicted mask [4, H, W] or [H, W]
            save_path: Output image path
            title: Optional figure title
            show: Whether to display the figure interactively
        """
        modal_names = ["T1c", "FLAIR", "T2"]
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        for row in range(3):
            gt = gt_img[row]
            pred = pred_img[row]

            if gt.min() < 0:
                gt = (gt + 1.0) / 2.0
            if pred.min() < 0:
                pred = (pred + 1.0) / 2.0

            gt = np.clip(gt, 0, 1)
            pred = np.clip(pred, 0, 1)
            err = np.abs(pred - gt)

            axes[row, 0].imshow(gt, cmap="gray")
            axes[row, 0].set_title(f"GT {modal_names[row]}")
            axes[row, 1].imshow(pred, cmap="gray")
            axes[row, 1].set_title(f"Pred {modal_names[row]}")
            axes[row, 2].imshow(err, cmap="magma")
            axes[row, 2].set_title(f"Abs Err {modal_names[row]}")

            for col in range(3):
                axes[row, col].axis("off")

        if pred_mask is not None:
            if pred_mask.ndim == 3:
                mask_vis = np.max(pred_mask, axis=0)
            else:
                mask_vis = pred_mask
            axes[0, 2].imshow(mask_vis, cmap="viridis", alpha=0.35)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.bg_color)

        if show:
            try:
                plt.show(block=False)
                plt.pause(1.0)
            except Exception:
                pass

        plt.close(fig)

    def show_inference_result(
        self,
        pred_img: np.ndarray,
        pred_mask: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Show inference result preview in an interactive matplotlib window.
        Falls back silently in headless environments.
        """
        modal_names = ["T1c", "FLAIR", "T2"]
        cols = 4 if pred_mask is not None else 3
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

        if cols == 3:
            axes = np.array(axes)

        for i in range(3):
            img = pred_img[i]
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = np.clip(img, 0, 1)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(modal_names[i])
            axes[i].axis("off")

        if pred_mask is not None:
            if pred_mask.ndim == 3:
                mask = np.max(pred_mask, axis=0)
            else:
                mask = pred_mask
            axes[3].imshow(mask, cmap="viridis")
            axes[3].set_title("Mask")
            axes[3].axis("off")

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        try:
            plt.show(block=False)
            plt.pause(1.5)
        except Exception:
            pass
        finally:
            plt.close(fig)

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_visualization_results(
    session_path: str,
    file_prefix: str,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    COLOR_MAP: Dict[str, str]
) -> None:
    """
    Save all visualization results for a session.
    
    Args:
        session_path: Path to save session results
        file_prefix: Prefix for saved files
        images: Dictionary containing different image arrays 3, h, w
        masks: Dictionary containing different mask arrays  4, h, w
    """
    create_directory(session_path)
    
    # Save ground truth and predicted masks
    for mask_name, masks_4session in masks.items():
        for j in range(4):
            mask_data = masks_4session[j, :, :].astype(float)
            mask_image = Visualizer(COLOR_MAP).to_pil(mask_data)
            mask_image.save(os.path.join(session_path, f"{file_prefix}-mask-sess{j}-{mask_name}.png"))
        
    # Save images with overlays and contours
    for img_name, img_3modal in images.items():
        # print(f'image: {img_name}, img_3modal.shape: {img_3modal.shape}')
        for i in range(3):  # For each modality
            img_data = img_3modal[i, :, :].astype(float)
            base_image = Visualizer(COLOR_MAP).to_pil(img_data)
            
            # Save original image
            base_image.save(os.path.join(session_path, f"{file_prefix}-image-modal{i}-{img_name}.png"))
        
        # Create and save overlay
        if 'ref_mask' in masks and 'pred_mask' in masks:
            overlay_image = Visualizer(COLOR_MAP).overlay_maps(base_image, Visualizer(COLOR_MAP).to_pil(masks['pred_mask'][3]), Visualizer(COLOR_MAP).to_pil(masks['ref_mask'][2]))
            overlay_image.save(os.path.join(session_path, f"{file_prefix}-{img_name}_overlay.png"))
        
        # Create and save contour
        if 'gt_mask' in masks:
            contour_image = Visualizer(COLOR_MAP).draw_contour(base_image, Visualizer(COLOR_MAP).to_pil(masks['gt_mask'][3]))
            contour_image.save(os.path.join(session_path, f"{file_prefix}-{img_name}_contour_gt.png"))
        if 'pred_mask' in masks:
            contour_image = Visualizer(COLOR_MAP).draw_contour(base_image, Visualizer(COLOR_MAP).to_pil(masks['pred_mask'][3]))
            contour_image.save(os.path.join(session_path, f"{file_prefix}-{img_name}_contour_pred.png"))
