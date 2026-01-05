"""
TryOn Backend - Face and Hair Compositor
Step 6: Composite user face AND hair onto try-on body result

Uses MediaPipe Face Detector (new tasks API) for head detection.
Uses OpenCV for blending.
Compatible with MediaPipe 0.10.30+
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime
from PIL import Image
from cloudinary_storage import CloudinaryStorage as StorageService

# MediaPipe new tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


@dataclass
class CompositeResult:
    """Result of face+hair compositing"""
    success: bool
    final_image_path: Optional[str] = None
    final_image_url: Optional[str] = None  # Cloud Storage URL
    final_id: Optional[str] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_image_path": self.final_image_path,
            "final_image_url": self.final_image_url,
            "final_id": self.final_id,
            "processed_at": self.processed_at,
            "error": self.error
        }


class FaceHairCompositor:
    """
    Composites user face AND hair onto try-on body image.
    
    Layers (bottom to top):
    1. Try-on body (base)
    2. Face overlay (aligned to head)
    3. Hair overlay (above face)
    """
    
    # BlazeFace Short Range model for face detection
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_PATH = "./models/blaze_face_short_range.tflite"
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        
        # Download model if not exists
        self._ensure_model()
        
        # Initialize Face Detector
        base_options = mp_tasks.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
    
    def _ensure_model(self):
        """Download model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            import urllib.request
            print(f"Downloading face detector model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")
    
    def composite(
        self,
        user_id: str,
        tryon_body_path: str,
        face_path: str,
        hair_path: str,
        skin_tone_rgb: List[int]
    ) -> CompositeResult:
        """
        Composite user face and hair onto try-on body.
        
        Args:
            user_id: User identifier
            tryon_body_path: Path to try-on body image
            face_path: Path to extracted face image
            hair_path: Path to extracted hair image (RGBA)
            skin_tone_rgb: User's skin tone [R, G, B]
            
        Returns:
            CompositeResult with path to final image
        """
        try:
            # Load images
            tryon_body = cv2.imread(tryon_body_path)
            face_img = cv2.imread(face_path, cv2.IMREAD_UNCHANGED)
            hair_img = cv2.imread(hair_path, cv2.IMREAD_UNCHANGED)
            
            if tryon_body is None:
                return CompositeResult(
                    success=False,
                    error=f"Could not load try-on body: {tryon_body_path}"
                )
            
            if face_img is None:
                return CompositeResult(
                    success=False,
                    error=f"Could not load face: {face_path}"
                )
            
            # Detect head region in try-on body
            head_bbox = self._detect_head(tryon_body)
            
            if head_bbox is None:
                # Fallback: assume head is top 25% of image
                h, w = tryon_body.shape[:2]
                head_bbox = (w // 4, 0, w // 2, int(h * 0.25))
            
            # Start with try-on body
            result = tryon_body.copy()
            
            # Layer 1: Blend face onto head region
            result = self._blend_face(result, face_img, head_bbox)
            
            # Layer 2: Overlay hair (if available)
            if hair_img is not None:
                result = self._overlay_hair(result, hair_img, head_bbox)
            
            # Adjust skin tone in visible areas
            result = self._adjust_skin_tone(result, skin_tone_rgb, head_bbox)
            
            # Save result
            final_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self._save_result(result, user_id, final_id)
            
            # Upload to Cloud Storage
            final_url = StorageService.upload_final(user_id, final_id, output_path)
            
            return CompositeResult(
                success=True,
                final_image_path=output_path,
                final_image_url=final_url,
                final_id=final_id,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return CompositeResult(
                success=False,
                error=str(e)
            )
    
    def _detect_head(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect head region using MediaPipe Face Detector."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect faces
        detection_result = self.face_detector.detect(mp_image)
        
        if not detection_result.detections:
            return None
        
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        
        x = bbox.origin_x
        y = bbox.origin_y
        width = bbox.width
        height = bbox.height
        
        # Expand for better coverage
        padding = int(min(width, height) * 0.25)
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        return (x, y, width, height)
    
    def _blend_face(
        self,
        body: np.ndarray,
        face: np.ndarray,
        head_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Blend face onto body with feathered edges."""
        x, y, width, height = head_bbox
        result = body.copy()
        
        # Resize face to match head region
        if face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
            face[:, :, 3] = 255
        
        face_resized = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create elliptical mask for face
        mask = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, height // 2)
        axes = (int(width * 0.42), int(height * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Feather edges
        mask = cv2.GaussianBlur(mask, (25, 25), 12)
        
        # Apply blending
        face_bgr = face_resized[:, :, :3]
        for c in range(3):
            body_region = result[y:y+height, x:x+width, c].astype(np.float32)
            face_channel = face_bgr[:, :, c].astype(np.float32)
            blended = body_region * (1 - mask) + face_channel * mask
            result[y:y+height, x:x+width, c] = blended.astype(np.uint8)
        
        return result
    
    def _overlay_hair(
        self,
        body: np.ndarray,
        hair: np.ndarray,
        head_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Overlay hair onto body above face layer."""
        x, y, width, height = head_bbox
        result = body.copy()
        body_h, body_w = body.shape[:2]
        
        # Ensure hair has alpha channel
        if hair.shape[2] == 3:
            return result  # No alpha, skip hair overlay
        
        # Calculate hair region (slightly larger and higher than face)
        hair_scale = 1.4
        hair_width = int(width * hair_scale)
        hair_height = int(height * hair_scale)
        
        # Position hair above face
        hair_x = x - int((hair_width - width) / 2)
        hair_y = y - int(height * 0.3)  # Move up for hair above forehead
        
        # Clamp to image bounds
        hair_x = max(0, hair_x)
        hair_y = max(0, hair_y)
        
        # Resize hair
        hair_resized = cv2.resize(hair, (hair_width, hair_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Get alpha channel
        alpha = hair_resized[:, :, 3].astype(np.float32) / 255.0
        
        # Feather hair edges for smoother blend
        alpha = cv2.GaussianBlur(alpha, (11, 11), 5)
        
        # Calculate valid overlay region
        end_x = min(hair_x + hair_width, body_w)
        end_y = min(hair_y + hair_height, body_h)
        
        actual_width = end_x - hair_x
        actual_height = end_y - hair_y
        
        if actual_width <= 0 or actual_height <= 0:
            return result
        
        # Crop hair to fit
        hair_crop = hair_resized[:actual_height, :actual_width, :3]
        alpha_crop = alpha[:actual_height, :actual_width]
        
        # Overlay with alpha blending
        for c in range(3):
            body_region = result[hair_y:end_y, hair_x:end_x, c].astype(np.float32)
            hair_channel = hair_crop[:, :, c].astype(np.float32)
            blended = body_region * (1 - alpha_crop) + hair_channel * alpha_crop
            result[hair_y:end_y, hair_x:end_x, c] = blended.astype(np.uint8)
        
        return result
    
    def _adjust_skin_tone(
        self,
        image: np.ndarray,
        skin_tone_rgb: List[int],
        head_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Adjust neck/visible skin to match user skin tone."""
        x, y, width, height = head_bbox
        result = image.copy()
        
        # Neck region
        neck_y_start = y + height
        neck_y_end = min(image.shape[0], neck_y_start + int(height * 0.3))
        neck_x_start = x + int(width * 0.25)
        neck_x_end = x + int(width * 0.75)
        
        if neck_y_end <= neck_y_start or neck_x_end <= neck_x_start:
            return result
        
        neck_region = result[neck_y_start:neck_y_end, neck_x_start:neck_x_end]
        if neck_region.size == 0:
            return result
        
        # Color adjustment
        current_avg = np.mean(neck_region, axis=(0, 1))
        target = np.array([skin_tone_rgb[2], skin_tone_rgb[1], skin_tone_rgb[0]])
        adjustment = (target - current_avg) * 0.3
        
        # Gradient for smooth transition
        h = neck_y_end - neck_y_start
        w = neck_x_end - neck_x_start
        gradient = np.linspace(1, 0, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w))
        
        for c in range(3):
            channel = neck_region[:, :, c].astype(np.float32)
            channel += adjustment[c] * gradient
            neck_region[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        result[neck_y_start:neck_y_end, neck_x_start:neck_x_end] = neck_region
        return result
    
    def _save_result(self, image: np.ndarray, user_id: str, final_id: str) -> str:
        """Save the final composite image."""
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        filename = f"final_tryon_{final_id}.png"
        output_path = os.path.join(user_dir, filename)
        
        cv2.imwrite(output_path, image)
        return output_path
    
    def close(self):
        """Release resources."""
        self.face_detector.close()


# === API Functions ===

def composite_face_and_hair(
    user_id: str,
    tryon_body_path: str,
    face_path: str,
    hair_path: str,
    skin_tone_rgb: List[int],
    output_dir: str = "./output"
) -> dict:
    """
    Create final try-on image with user face and hair.
    """
    compositor = FaceHairCompositor(output_dir=output_dir)
    try:
        result = compositor.composite(
            user_id, tryon_body_path, face_path, hair_path, skin_tone_rgb
        )
        return result.to_dict()
    finally:
        compositor.close()


# Backward compatible function
def composite_face(
    user_id: str,
    tryon_body_path: str,
    face_path: str,
    skin_tone_rgb: List[int],
    output_dir: str = "./output"
) -> dict:
    """Backward compatible - face only (no hair)."""
    # Try to find hair image
    hair_path = os.path.join(output_dir, f"{user_id}_hair.png")
    if not os.path.exists(hair_path):
        hair_path = None
    
    compositor = FaceHairCompositor(output_dir=output_dir)
    try:
        result = compositor.composite(
            user_id, tryon_body_path, face_path, 
            hair_path if hair_path else face_path,  # Use face as dummy if no hair
            skin_tone_rgb
        )
        return result.to_dict()
    finally:
        compositor.close()


# === CLI ===

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 6:
        print("Usage: python face_compositor.py <user_id> <tryon_body> <face> <hair> <r> <g> <b>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    tryon_path = sys.argv[2]
    face_path = sys.argv[3]
    hair_path = sys.argv[4]
    skin = [int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])]
    
    result = composite_face_and_hair(user_id, tryon_path, face_path, hair_path, skin)
    print(json.dumps(result, indent=2))
