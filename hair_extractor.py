"""
TryOn Backend - Hair Extraction Service
Step 2A: Extract hair mask from selfie image

Uses MediaPipe Image Segmenter (new tasks API) with selfie multiclass model.
Compatible with MediaPipe 0.10.30+
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from cloudinary_storage import CloudinaryStorage as StorageService

# MediaPipe new tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


@dataclass
class HairExtractionResult:
    """Result of hair extraction"""
    success: bool
    hair_mask_path: Optional[str] = None
    hair_mask_url: Optional[str] = None  # Cloud Storage URL
    hair_image_path: Optional[str] = None
    hair_image_url: Optional[str] = None  # Cloud Storage URL
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "hair_mask_path": self.hair_mask_path,
            "hair_mask_url": self.hair_mask_url,
            "hair_image_path": self.hair_image_path,
            "hair_image_url": self.hair_image_url,
            "processed_at": self.processed_at,
            "error": self.error
        }


class HairExtractor:
    """
    Extracts hair mask from a selfie image.
    
    Uses MediaPipe Image Segmenter with selfie multiclass model
    to segment hair from the image.
    """
    
    # Model URLs - selfie multiclass segmentation
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    MODEL_PATH = "./models/selfie_multiclass.tflite"
    
    # Segmentation class indices (for multiclass model)
    # 0: background, 1: hair, 2: body-skin, 3: face-skin, 4: clothes, 5: others
    HAIR_CLASS_INDEX = 1
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the hair extractor.
        
        Args:
            output_dir: Directory to save extracted hair images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        
        # Download model if not exists
        self._ensure_model()
        
        # Create Image Segmenter
        base_options = mp_tasks.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            output_confidence_masks=False
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
    
    def _ensure_model(self):
        """Download model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            import urllib.request
            print(f"Downloading selfie segmentation model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")
    
    def extract(self, image_path: str, user_id: str) -> HairExtractionResult:
        """
        Extract hair mask from a selfie image.
        
        Args:
            image_path: Path to the input selfie image
            user_id: Unique user identifier for output file naming
            
        Returns:
            HairExtractionResult with hair mask path
        """
        try:
            # Load image
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                return HairExtractionResult(
                    success=False,
                    error=f"Could not load image: {image_path}"
                )
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            h, w = image_cv.shape[:2]
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Perform segmentation
            segmentation_result = self.segmenter.segment(mp_image)
            
            # Get category mask
            category_mask = segmentation_result.category_mask
            if category_mask is None:
                return HairExtractionResult(
                    success=False,
                    error="Could not segment image"
                )
            
            # Convert to numpy array
            mask_array = category_mask.numpy_view()
            
            # Extract hair mask (class index 1 = hair)
            hair_mask = (mask_array == self.HAIR_CLASS_INDEX).astype(np.uint8) * 255
            
            # Resize mask to original image size if needed
            if hair_mask.shape[:2] != (h, w):
                hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Clean up hair mask
            hair_mask = self._clean_hair_mask(hair_mask)
            
            # Check if any hair was detected (lowered threshold for better sensitivity)
            hair_pixels = np.sum(hair_mask > 0)
            print(f"Hair pixels detected: {hair_pixels}")
            
            if hair_pixels < 100:  # Lowered from 1000 to be more sensitive
                return HairExtractionResult(
                    success=False,
                    error=f"No significant hair detected ({hair_pixels} pixels)"
                )
            
            # Save hair mask
            hair_mask_path = self._save_hair_mask(hair_mask, user_id)
            
            # Save hair image (with transparency)
            hair_image_path = self._save_hair_image(image_cv, hair_mask, user_id)
            
            # Upload to Cloud Storage
            hair_mask_url = StorageService.upload_hair_mask(user_id, hair_mask_path)
            hair_image_url = StorageService.upload_hair(user_id, hair_image_path)
            
            return HairExtractionResult(
                success=True,
                hair_mask_path=hair_mask_path,
                hair_mask_url=hair_mask_url,
                hair_image_path=hair_image_path,
                hair_image_url=hair_image_url,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return HairExtractionResult(
                success=False,
                error=str(e)
            )
    
    def _clean_hair_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up hair mask with morphological operations.
        """
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Keep only largest connected component (main hair region)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest_label) * 255).astype(np.uint8)
        
        return mask
    
    def _save_hair_mask(self, mask: np.ndarray, user_id: str) -> str:
        """Save the binary hair mask."""
        filename = f"{user_id}_hair_mask.png"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, mask)
        return path
    
    def _save_hair_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        user_id: str
    ) -> str:
        """Save hair image with transparency (RGBA)."""
        h, w = image.shape[:2]
        
        # Create RGBA image
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image  # BGR channels
        rgba[:, :, 3] = mask    # Alpha channel
        
        filename = f"{user_id}_hair.png"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, rgba)
        return path
    
    def close(self):
        """Release MediaPipe resources."""
        self.segmenter.close()


# === API Functions ===

def extract_hair(
    image_path: str, 
    user_id: str, 
    output_dir: str = "./output"
) -> dict:
    """
    Extract hair mask from selfie.
    
    Args:
        image_path: Path to selfie image
        user_id: Unique user identifier
        output_dir: Directory for output files
        
    Returns:
        Dictionary with extraction results
    """
    extractor = HairExtractor(output_dir=output_dir)
    try:
        result = extractor.extract(image_path, user_id)
        return result.to_dict()
    finally:
        extractor.close()


# === CLI for testing ===

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python hair_extractor.py <image_path> <user_id>")
        print("Example: python hair_extractor.py selfie.jpg user123")
        sys.exit(1)
    
    image_path = sys.argv[1]
    user_id = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./output"
    
    print(f"Extracting hair from: {image_path}")
    print(f"User ID: {user_id}")
    print("-" * 40)
    
    result = extract_hair(image_path, user_id, output_dir)
    
    print(json.dumps(result, indent=2))
