"""
TryOn Backend - Face and Skin Extraction Service
Step 2: Extract face and skin tone from selfie image

Uses MediaPipe Face Landmarker (new tasks API) for face detection and landmark extraction.
Compatible with MediaPipe 0.10.30+
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from datetime import datetime
import json
from cloudinary_storage import CloudinaryStorage as StorageService

# MediaPipe new tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


@dataclass
class FaceExtractionResult:
    """Result of face and skin extraction"""
    success: bool
    face_image_path: Optional[str] = None  # Local path (for backward compat)
    face_image_url: Optional[str] = None   # Cloud Storage URL
    skin_tone_rgb: Optional[List[int]] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "face_image_path": self.face_image_path,
            "face_image_url": self.face_image_url,
            "skin_tone_rgb": self.skin_tone_rgb,
            "processed_at": self.processed_at,
            "error": self.error
        }


class FaceExtractor:
    """
    Extracts face and skin tone from a selfie image.
    
    Uses MediaPipe Face Landmarker (tasks API) for face detection.
    """
    
    # MediaPipe model asset paths
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "./models/face_landmarker.task"
    
    # Output dimensions
    FACE_OUTPUT_SIZE = 512
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the face extractor.
        
        Args:
            output_dir: Directory to save extracted face images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        
        # Download model if not exists
        self._ensure_model()
        
        # Create FaceLandmarker
        base_options = mp_tasks.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    def _ensure_model(self):
        """Download model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            import urllib.request
            print(f"Downloading face landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")
    
    def extract(self, image_path: str, user_id: str) -> FaceExtractionResult:
        """
        Extract face and skin tone from a selfie image.
        
        Args:
            image_path: Path to the input selfie image
            user_id: Unique user identifier for output file naming
            
        Returns:
            FaceExtractionResult with face image path and skin tone
        """
        try:
            # Load image with OpenCV first
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                return FaceExtractionResult(
                    success=False,
                    error=f"Could not load image: {image_path}"
                )
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            h, w = image_cv.shape[:2]
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect face landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return FaceExtractionResult(
                    success=False,
                    error="No face detected in the image"
                )
            
            face_landmarks = detection_result.face_landmarks[0]
            
            # Get face bounding box from landmarks
            face_bbox = self._get_face_bbox(face_landmarks, w, h)
            
            # Crop and save face
            face_image_path = self._crop_and_save_face(
                image_cv, face_bbox, user_id
            )
            
            # Upload to Cloud Storage
            face_image_url = StorageService.upload_face(user_id, face_image_path)
            
            # Extract skin tone
            skin_tone_rgb = self._extract_skin_tone(
                image_rgb, face_landmarks, w, h
            )
            
            return FaceExtractionResult(
                success=True,
                face_image_path=face_image_path,
                face_image_url=face_image_url,
                skin_tone_rgb=skin_tone_rgb,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return FaceExtractionResult(
                success=False,
                error=str(e)
            )
    
    def _get_face_bbox(
        self, 
        landmarks, 
        image_width: int, 
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate tight face bounding box using specific facial landmarks.
        Uses key points: forehead, chin, left ear, right ear for precise face-only crop.
        
        Returns: (x, y, width, height)
        """
        # MediaPipe Face Mesh key landmark indices for face boundary
        # Forehead top: 10 (center top of forehead)
        # Chin bottom: 152 (bottom of chin)
        # Left cheek: 234 (outer left face)
        # Right cheek: 454 (outer right face)
        # Left eyebrow: 21 (outer left eyebrow)
        # Right eyebrow: 251 (outer right eyebrow)
        
        # Key landmarks for tight face crop
        FOREHEAD_IDX = 10      # Top of forehead
        CHIN_IDX = 152         # Bottom of chin
        LEFT_FACE_IDX = 234    # Left side of face
        RIGHT_FACE_IDX = 454   # Right side of face
        LEFT_EYEBROW_IDX = 21  # For vertical reference
        RIGHT_EYEBROW_IDX = 251
        
        # Get key points
        try:
            forehead = landmarks[FOREHEAD_IDX]
            chin = landmarks[CHIN_IDX]
            left_face = landmarks[LEFT_FACE_IDX]
            right_face = landmarks[RIGHT_FACE_IDX]
            
            # Calculate boundaries from key landmarks
            top_y = forehead.y * image_height
            bottom_y = chin.y * image_height
            left_x = left_face.x * image_width
            right_x = right_face.x * image_width
            
        except (IndexError, AttributeError):
            # Fallback to all landmarks if key indices fail
            xs = [lm.x * image_width for lm in landmarks]
            ys = [lm.y * image_height for lm in landmarks]
            left_x, right_x = min(xs), max(xs)
            top_y, bottom_y = min(ys), max(ys)
        
        # Calculate dimensions
        face_width = right_x - left_x
        face_height = bottom_y - top_y
        
        # Add minimal padding (5% for clean edges, not background)
        padding_w = int(face_width * 0.05)
        padding_h = int(face_height * 0.08)  # Slightly more vertical for forehead
        
        # Final bounding box
        x = max(0, int(left_x - padding_w))
        y = max(0, int(top_y - padding_h))
        w = min(image_width - x, int(face_width + 2 * padding_w))
        h = min(image_height - y, int(face_height + 2 * padding_h))
        
        return (x, y, w, h)
    
    def _crop_and_save_face(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        user_id: str
    ) -> str:
        """
        Crop face region and save as square image.
        """
        x, y, w, h = bbox
        
        # Crop face region
        face_crop = image[y:y+h, x:x+w].copy()
        
        # Resize to output size (square)
        face_resized = cv2.resize(
            face_crop, 
            (self.FACE_OUTPUT_SIZE, self.FACE_OUTPUT_SIZE),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Save
        filename = f"{user_id}_face.png"
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, face_resized)
        
        return output_path
    
    def _extract_skin_tone(
        self, 
        image_rgb: np.ndarray, 
        landmarks, 
        w: int, 
        h: int
    ) -> List[int]:
        """
        Extract average skin tone from cheek/forehead regions.
        """
        # Key landmark indices for skin sampling (cheeks, forehead)
        # MediaPipe face landmarks indices
        CHEEK_LEFT_INDICES = [50, 101, 116]  # Left cheek area
        CHEEK_RIGHT_INDICES = [280, 330, 345]  # Right cheek area
        FOREHEAD_INDICES = [10, 151, 9]  # Forehead area
        
        skin_pixels = []
        
        all_indices = CHEEK_LEFT_INDICES + CHEEK_RIGHT_INDICES + FOREHEAD_INDICES
        
        for idx in all_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                
                # Sample a small region around the point
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            pixel = image_rgb[ny, nx]
                            skin_pixels.append(pixel)
        
        if not skin_pixels:
            return [200, 160, 140]  # Default skin tone
        
        # Calculate average
        skin_pixels = np.array(skin_pixels)
        avg_rgb = np.mean(skin_pixels, axis=0).astype(int)
        
        return avg_rgb.tolist()
    
    def close(self):
        """Release resources."""
        self.face_landmarker.close()


# === API Function ===

def extract_face_and_skin(
    image_path: str, 
    user_id: str, 
    output_dir: str = "./output"
) -> dict:
    """
    Main API function to extract face and skin tone.
    
    Args:
        image_path: Path to selfie image
        user_id: Unique user identifier
        output_dir: Directory for output files
        
    Returns:
        Dictionary with extraction results
    """
    extractor = FaceExtractor(output_dir=output_dir)
    try:
        result = extractor.extract(image_path, user_id)
        return result.to_dict()
    finally:
        extractor.close()


# === CLI for testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python face_extractor.py <image_path> <user_id>")
        print("Example: python face_extractor.py selfie.jpg user123")
        sys.exit(1)
    
    image_path = sys.argv[1]
    user_id = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./output"
    
    print(f"Extracting face and skin from: {image_path}")
    print(f"User ID: {user_id}")
    print("-" * 40)
    
    result = extract_face_and_skin(image_path, user_id, output_dir)
    
    print(json.dumps(result, indent=2))
