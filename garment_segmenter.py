"""
TryOn Backend - Garment Segmentation
Step 4: Extract clean garment from outfit image

Uses rembg (backed by U2-Net) for robust background removal.
Note: SAM requires significant GPU memory; rembg is more practical for deployment.
"""

import os
import cv2
import numpy as np
import requests
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
from cloudinary_storage import CloudinaryStorage as StorageService

# Use rembg for background removal (U2-Net based, works well for clothing)
# For SAM, you'd need: from segment_anything import sam_model_registry, SamPredictor
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Run: pip install rembg")


@dataclass
class GarmentSegmentationResult:
    """Result of garment segmentation"""
    success: bool
    garment_image_path: Optional[str] = None
    garment_image_url: Optional[str] = None  # Firebase Storage URL
    garment_id: Optional[str] = None
    garment_type: Optional[str] = None  # 'top', 'bottom', 'dress'
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "garment_image_path": self.garment_image_path,
            "garment_image_url": self.garment_image_url,
            "garment_id": self.garment_id,
            "garment_type": self.garment_type,
            "processed_at": self.processed_at,
            "error": self.error
        }


class GarmentSegmenter:
    """
    Segments garments from outfit images.
    
    Supports:
    - File upload
    - URL download
    
    Outputs transparent PNG with just the garment.
    """
    
    # Standard output dimensions
    OUTPUT_SIZE = (768, 1024)
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the garment segmenter.
        
        Args:
            output_dir: Directory to save segmented garment images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def segment_from_file(
        self, 
        image_path: str, 
        user_id: str,
        garment_id: Optional[str] = None
    ) -> GarmentSegmentationResult:
        """
        Segment garment from a local image file.
        
        Args:
            image_path: Path to the outfit image
            user_id: User identifier
            garment_id: Optional garment identifier for naming
            
        Returns:
            GarmentSegmentationResult
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return GarmentSegmentationResult(
                    success=False,
                    error=f"Could not load image: {image_path}"
                )
            
            return self._process_image(image, user_id, garment_id)
            
        except Exception as e:
            return GarmentSegmentationResult(
                success=False,
                error=str(e)
            )
    
    def segment_from_url(
        self, 
        image_url: str, 
        user_id: str,
        garment_id: Optional[str] = None
    ) -> GarmentSegmentationResult:
        """
        Download and segment garment from a URL.
        
        Args:
            image_url: URL of the outfit image
            user_id: User identifier
            garment_id: Optional garment identifier for naming
            
        Returns:
            GarmentSegmentationResult
        """
        try:
            # Validate URL
            parsed = urlparse(image_url)
            if not parsed.scheme in ['http', 'https']:
                return GarmentSegmentationResult(
                    success=False,
                    error="Invalid URL scheme. Use http or https."
                )
            
            # Download image
            response = requests.get(image_url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                return GarmentSegmentationResult(
                    success=False,
                    error=f"Failed to download image: HTTP {response.status_code}"
                )
            
            # Load image from bytes
            image_bytes = BytesIO(response.content)
            pil_image = Image.open(image_bytes)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return self._process_image(image, user_id, garment_id)
            
        except requests.Timeout:
            return GarmentSegmentationResult(
                success=False,
                error="Download timeout"
            )
        except Exception as e:
            return GarmentSegmentationResult(
                success=False,
                error=str(e)
            )
    
    def _process_image(
        self, 
        image: np.ndarray, 
        user_id: str,
        garment_id: Optional[str] = None
    ) -> GarmentSegmentationResult:
        """
        Process image: resize, segment, detect type, save.
        """
        if not REMBG_AVAILABLE:
            return GarmentSegmentationResult(
                success=False,
                error="rembg not installed. Run: pip install rembg"
            )
        
        try:
            # Resize to standard dimensions
            resized = self._resize_image(image)
            
            # Convert to PIL for rembg
            pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            
            # Remove background (returns RGBA image)
            segmented = remove(pil_image)
            
            # Detect garment type based on shape analysis
            garment_type = self._detect_garment_type(segmented)
            
            # Save the result
            actual_garment_id = garment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            garment_path = self._save_garment(segmented, user_id, actual_garment_id)
            
            # Upload to Firebase Storage
            garment_url = StorageService.upload_garment(user_id, actual_garment_id, garment_path)
            
            return GarmentSegmentationResult(
                success=True,
                garment_image_path=garment_path,
                garment_image_url=garment_url,
                garment_id=actual_garment_id,
                garment_type=garment_type,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return GarmentSegmentationResult(
                success=False,
                error=str(e)
            )
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to standard dimensions while maintaining aspect ratio.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.OUTPUT_SIZE
        
        # Calculate scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas and center
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _detect_garment_type(self, image: Image.Image) -> str:
        """
        Detect garment type based on shape analysis.
        
        Heuristic: Analyze the bounding box aspect ratio and position
        of non-transparent pixels.
        
        Returns: 'top', 'bottom', or 'dress'
        """
        # Convert to numpy and get alpha channel
        img_array = np.array(image)
        if img_array.shape[2] < 4:
            return 'top'  # Default if no alpha
        
        alpha = img_array[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return 'top'  # Default if no content
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        height = y_max - y_min
        width = x_max - x_min
        img_height = alpha.shape[0]
        
        # Calculate metrics
        aspect_ratio = height / max(width, 1)
        vertical_position = y_min / img_height  # 0 = top, 1 = bottom
        coverage = height / img_height
        
        # Classification heuristics
        if coverage > 0.7:
            # Covers most of the image height -> dress or full outfit
            return 'dress'
        elif vertical_position < 0.3 and aspect_ratio < 1.5:
            # Starts near top, not too tall -> top
            return 'top'
        elif vertical_position > 0.4:
            # Starts in lower half -> bottom
            return 'bottom'
        elif aspect_ratio > 1.8:
            # Very tall and thin -> probably pants/bottom
            return 'bottom'
        else:
            # Default to top
            return 'top'
    
    def _save_garment(
        self, 
        image: Image.Image, 
        user_id: str,
        garment_id: Optional[str] = None
    ) -> str:
        """
        Save the segmented garment as transparent PNG.
        """
        # Create user directory
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate filename
        if garment_id:
            filename = f"garment_{garment_id}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"garment_{timestamp}.png"
        
        output_path = os.path.join(user_dir, filename)
        
        # Save as PNG with transparency
        image.save(output_path, "PNG")
        
        return output_path


# === API Functions ===

def segment_garment_from_file(
    image_path: str,
    user_id: str,
    garment_id: Optional[str] = None,
    output_dir: str = "./output"
) -> dict:
    """
    Segment garment from a local file.
    """
    segmenter = GarmentSegmenter(output_dir=output_dir)
    result = segmenter.segment_from_file(image_path, user_id, garment_id)
    return result.to_dict()


def segment_garment_from_url(
    image_url: str,
    user_id: str,
    garment_id: Optional[str] = None,
    output_dir: str = "./output"
) -> dict:
    """
    Download and segment garment from a URL.
    """
    segmenter = GarmentSegmenter(output_dir=output_dir)
    result = segmenter.segment_from_url(image_url, user_id, garment_id)
    return result.to_dict()


# === CLI for testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python garment_segmenter.py file <image_path> <user_id>")
        print("  python garment_segmenter.py url <image_url> <user_id>")
        sys.exit(1)
    
    mode = sys.argv[1]
    source = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else "test_user"
    
    import json
    
    if mode == "file":
        print(f"Segmenting from file: {source}")
        result = segment_garment_from_file(source, user_id)
    elif mode == "url":
        print(f"Segmenting from URL: {source}")
        result = segment_garment_from_url(source, user_id)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    print(json.dumps(result, indent=2))
