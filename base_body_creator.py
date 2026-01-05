"""
TryOn Backend - Base Body Creator
Step 3: Create and cache personalized base body image

Loads pre-existing base body templates and personalizes them based on user profile.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
import json
from cloudinary_storage import CloudinaryStorage as StorageService


@dataclass
class BaseBodyResult:
    """Result of base body creation"""
    success: bool
    base_body_image_path: Optional[str] = None
    base_body_image_url: Optional[str] = None  # Firebase Storage URL
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "base_body_image_path": self.base_body_image_path,
            "base_body_image_url": self.base_body_image_url,
            "processed_at": self.processed_at,
            "error": self.error
        }


class BaseBodyCreator:
    """
    Creates a personalized base body image from templates.
    
    Templates are stored at: assets/base_bodies/{gender}/{body_build}.png
    
    This runs ONCE per user during onboarding.
    """
    
    # Standard output dimensions
    OUTPUT_HEIGHT = 1024
    OUTPUT_WIDTH = 768
    
    # Height range for scaling (cm)
    MIN_HEIGHT = 140
    MAX_HEIGHT = 210
    REFERENCE_HEIGHT = 170  # Height at which template is 1:1 scale
    
    def __init__(
        self, 
        templates_dir: str = "./assets/base_bodies",
        output_dir: str = "./output"
    ):
        """
        Initialize the base body creator.
        
        Args:
            templates_dir: Directory containing base body templates
            output_dir: Directory to save generated base body images
        """
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create(
        self,
        user_id: str,
        gender: str,
        body_build: str,
        height_cm: int
    ) -> BaseBodyResult:
        """
        Create a personalized base body image.
        
        Args:
            user_id: Unique user identifier
            gender: 'male', 'female', or 'other'
            body_build: 'slim', 'regular', 'chubby', 'muscular'
            height_cm: User's height in centimeters
            
        Returns:
            BaseBodyResult with path to created image
        """
        try:
            # Load template
            template_path = self._get_template_path(gender, body_build)
            template = cv2.imread(template_path)
            
            if template is None:
                return BaseBodyResult(
                    success=False,
                    error=f"Template not found: {template_path}"
                )
            
            # Process the template
            processed = self._process_template(template, height_cm)
            
            # Normalize lighting and contrast
            processed = self._normalize_lighting(processed)
            
            # Add subtle noise for photorealism
            processed = self._add_subtle_noise(processed)
            
            # Save the result
            output_path = self._save_base_body(processed, user_id)
            
            # Upload to Firebase Storage
            base_body_url = StorageService.upload_base_body(user_id, output_path)
            
            return BaseBodyResult(
                success=True,
                base_body_image_path=output_path,
                base_body_image_url=base_body_url,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return BaseBodyResult(
                success=False,
                error=str(e)
            )
    
    def _get_template_path(self, gender: str, body_build: str) -> str:
        """Get the path to the appropriate template."""
        # Normalize gender for file path
        gender_folder = gender.lower()
        if gender_folder not in ['male', 'female']:
            gender_folder = 'neutral'  # Fallback for 'other'
        
        # Build path
        template_path = os.path.join(
            self.templates_dir,
            gender_folder,
            f"{body_build}.png"
        )
        
        # Fallback to regular if specific build not found
        if not os.path.exists(template_path):
            fallback_path = os.path.join(
                self.templates_dir,
                gender_folder,
                "regular.png"
            )
            if os.path.exists(fallback_path):
                return fallback_path
        
        return template_path
    
    def _process_template(
        self, 
        template: np.ndarray, 
        height_cm: int
    ) -> np.ndarray:
        """
        Process template: resize and scale based on height.
        
        Args:
            template: Original template image
            height_cm: User's height in cm
            
        Returns:
            Processed image at OUTPUT_HEIGHT x OUTPUT_WIDTH
        """
        h, w = template.shape[:2]
        
        # Calculate scale factor based on height
        # Taller users get slightly stretched body, shorter get compressed
        height_factor = self._calculate_height_factor(height_cm)
        
        # First, resize to fit within output dimensions
        aspect = w / h
        target_aspect = self.OUTPUT_WIDTH / self.OUTPUT_HEIGHT
        
        if aspect > target_aspect:
            # Template is wider - fit to width
            new_w = self.OUTPUT_WIDTH
            new_h = int(new_w / aspect)
        else:
            # Template is taller - fit to height
            new_h = self.OUTPUT_HEIGHT
            new_w = int(new_h * aspect)
        
        resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply height scaling (stretch/compress vertically)
        scaled_h = int(new_h * height_factor)
        scaled = cv2.resize(resized, (new_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create output canvas and center the image
        output = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = max(0, (self.OUTPUT_HEIGHT - scaled_h) // 2)
        x_offset = max(0, (self.OUTPUT_WIDTH - new_w) // 2)
        
        # Handle case where scaled image is larger than output
        src_y_start = max(0, (scaled_h - self.OUTPUT_HEIGHT) // 2)
        src_x_start = max(0, (new_w - self.OUTPUT_WIDTH) // 2)
        
        # Copy region that fits
        copy_h = min(scaled_h, self.OUTPUT_HEIGHT)
        copy_w = min(new_w, self.OUTPUT_WIDTH)
        
        output[y_offset:y_offset+copy_h, x_offset:x_offset+copy_w] = \
            scaled[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
        
        return output
    
    def _calculate_height_factor(self, height_cm: int) -> float:
        """
        Calculate vertical scaling factor based on height.
        
        Returns a factor between 0.9 (short) and 1.1 (tall).
        """
        # Clamp height to valid range
        height_cm = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, height_cm))
        
        # Calculate normalized position (0 = min, 1 = max)
        normalized = (height_cm - self.MIN_HEIGHT) / (self.MAX_HEIGHT - self.MIN_HEIGHT)
        
        # Map to scale factor (0.92 to 1.08)
        # Subtle scaling to avoid distortion
        scale_factor = 0.92 + (normalized * 0.16)
        
        return scale_factor
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting and contrast for consistent appearance.
        
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        for natural-looking normalization.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        
        # Merge channels back
        lab_normalized = cv2.merge([l_normalized, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def _add_subtle_noise(
        self, 
        image: np.ndarray, 
        noise_level: float = 3.0
    ) -> np.ndarray:
        """
        Add very subtle noise to improve photorealism.
        
        This helps the image blend better with IDM-VTON outputs
        which typically have some texture.
        
        Args:
            image: Input image
            noise_level: Standard deviation of Gaussian noise (default: 3.0)
            
        Returns:
            Image with subtle noise added
        """
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        
        # Add noise to image
        noisy = image.astype(np.float32) + noise
        
        # Clip to valid range and convert back
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def _save_base_body(self, image: np.ndarray, user_id: str) -> str:
        """Save the base body image to user-specific directory."""
        # Create user directory
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save image
        filename = "base_body.png"
        output_path = os.path.join(user_dir, filename)
        cv2.imwrite(output_path, image)
        
        return output_path


# === API Function ===

def create_base_body(
    user_id: str,
    gender: str,
    body_build: str,
    height_cm: int,
    templates_dir: str = "./assets/base_bodies",
    output_dir: str = "./output"
) -> dict:
    """
    Main API function to create a personalized base body image.
    
    Args:
        user_id: Unique user identifier
        gender: 'male', 'female', or 'other'
        body_build: 'slim', 'regular', 'chubby', 'muscular'
        height_cm: User's height in cm
        templates_dir: Directory containing base body templates
        output_dir: Directory for output files
        
    Returns:
        Dictionary with creation results
    """
    creator = BaseBodyCreator(
        templates_dir=templates_dir,
        output_dir=output_dir
    )
    result = creator.create(user_id, gender, body_build, height_cm)
    return result.to_dict()


# === CLI for testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python base_body_creator.py <user_id> <gender> <body_build> <height_cm>")
        print("Example: python base_body_creator.py user123 male regular 175")
        sys.exit(1)
    
    user_id = sys.argv[1]
    gender = sys.argv[2]
    body_build = sys.argv[3]
    height_cm = int(sys.argv[4])
    
    print(f"Creating base body for: {user_id}")
    print(f"  Gender: {gender}")
    print(f"  Body Build: {body_build}")
    print(f"  Height: {height_cm}cm")
    print("-" * 40)
    
    result = create_base_body(user_id, gender, body_build, height_cm)
    
    print(json.dumps(result, indent=2))
