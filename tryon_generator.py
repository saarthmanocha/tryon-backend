"""
TryOn Backend - IDM-VTON Integration
Step 5: Generate dressed body using base body + garment

Uses fal.ai IDM-VTON API for virtual try-on inference.
"""

import os
import base64
import requests
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from PIL import Image
from io import BytesIO
from cloudinary_storage import CloudinaryStorage as StorageService


@dataclass
class TryOnResult:
    """Result of try-on generation"""
    success: bool
    tryon_image_path: Optional[str] = None
    tryon_image_url: Optional[str] = None  # Firebase Storage URL
    tryon_id: Optional[str] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "tryon_image_path": self.tryon_image_path,
            "tryon_image_url": self.tryon_image_url,
            "tryon_id": self.tryon_id,
            "processed_at": self.processed_at,
            "error": self.error
        }


class IDMVTONService:
    """
    Integrates with fal.ai IDM-VTON API for virtual try-on.
    
    Takes a base body image and a garment image,
    returns the dressed body image.
    """
    
    # fal.ai API endpoint
    API_ENDPOINT = "https://fal.run/fal-ai/idm-vton"
    
    def __init__(self, api_key: str, output_dir: str = "./output"):
        """
        Initialize the IDM-VTON service.
        
        Args:
            api_key: fal.ai API key
            output_dir: Directory to save generated images
        """
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_tryon(
        self,
        user_id: str,
        base_body_path: str,
        garment_path: str,
        garment_type: str = "upper_body"
    ) -> TryOnResult:
        """
        Generate a try-on image using IDM-VTON.
        
        Args:
            user_id: User identifier
            base_body_path: Path to the user's base body image
            garment_path: Path to the segmented garment image
            garment_type: 'upper_body', 'lower_body', or 'dresses'
            
        Returns:
            TryOnResult with path to generated image
        """
        try:
            # Validate inputs
            if not os.path.exists(base_body_path):
                return TryOnResult(
                    success=False,
                    error=f"Base body not found: {base_body_path}"
                )
            
            if not os.path.exists(garment_path):
                return TryOnResult(
                    success=False,
                    error=f"Garment not found: {garment_path}"
                )
            
            # Map garment type to API format
            category = self._map_garment_type(garment_type)
            
            # Upload images and get URLs (fal.ai needs URLs)
            human_image_url = self._upload_image(base_body_path)
            garment_image_url = self._upload_image(garment_path)
            
            # Call IDM-VTON API
            result_url = self._call_vton_api(
                human_image_url,
                garment_image_url,
                category
            )
            
            if not result_url:
                return TryOnResult(
                    success=False,
                    error="API did not return a result image"
                )
            
            # Download and save result
            tryon_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self._save_result(result_url, user_id, tryon_id)
            
            # Upload to Firebase Storage
            tryon_url = StorageService.upload_tryon(user_id, tryon_id, output_path)
            
            return TryOnResult(
                success=True,
                tryon_image_path=output_path,
                tryon_image_url=tryon_url,
                tryon_id=tryon_id,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return TryOnResult(
                success=False,
                error=str(e)
            )
    
    def _map_garment_type(self, garment_type: str) -> str:
        """Map our garment types to IDM-VTON categories."""
        mapping = {
            "top": "upper_body",
            "bottom": "lower_body",
            "dress": "dresses",
            "upper_body": "upper_body",
            "lower_body": "lower_body",
            "dresses": "dresses"
        }
        return mapping.get(garment_type.lower(), "upper_body")
    
    def _upload_image(self, image_path: str) -> str:
        """
        Upload image to imgbb and return URL.
        This is needed because fal.ai requires URLs, not files.
        """
        IMGBB_API_KEY = "44a16582c207a9f62871978831a25945"  # Free API key
        IMGBB_ENDPOINT = "https://api.imgbb.com/1/upload"
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Upload to imgbb
        response = requests.post(
            IMGBB_ENDPOINT,
            data={
                "key": IMGBB_API_KEY,
                "image": image_data,
                "expiration": 600  # 10 minutes
            },
            timeout=60
        )
        
        if response.status_code == 200 and response.json().get("success"):
            return response.json()["data"]["url"]
        
        raise Exception(f"Image upload failed: {response.text}")
    
    def _call_vton_api(
        self,
        human_image_url: str,
        garment_image_url: str,
        category: str
    ) -> Optional[str]:
        """
        Call the fal.ai IDM-VTON API.
        
        Returns the URL of the generated image.
        """
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "human_image_url": human_image_url,
            "garment_image_url": garment_image_url,
            "category": category
        }
        
        response = requests.post(
            self.API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            # fal.ai returns result in various formats
            result_url = (
                data.get("image", {}).get("url") or
                data.get("output", {}).get("image") or
                data.get("image") if isinstance(data.get("image"), str) else None
            )
            return result_url
        
        # Handle errors
        error_msg = "Unknown error"
        if response.status_code == 401:
            error_msg = "Invalid API key"
        elif response.status_code == 402:
            error_msg = "Insufficient credits"
        elif response.status_code == 429:
            error_msg = "Rate limit exceeded"
        else:
            try:
                error_msg = response.json().get("detail", response.text)
            except:
                error_msg = response.text
        
        raise Exception(f"API error ({response.status_code}): {error_msg}")
    
    def _save_result(self, image_url: str, user_id: str, tryon_id: str) -> str:
        """Download result image and save to user directory."""
        # Download image
        response = requests.get(image_url, timeout=60)
        if response.status_code != 200:
            raise Exception("Failed to download result image")
        
        # Create user directory
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save image
        filename = f"tryon_{tryon_id}.png"
        output_path = os.path.join(user_dir, filename)
        
        # Save as PNG
        image = Image.open(BytesIO(response.content))
        image.save(output_path, "PNG")
        
        return output_path



# === API Functions ===

# Global API key (set from environment or config)
_api_key: Optional[str] = None
# Mock mode flag
_mock_mode: bool = False

# Mock result image URL (a sample dressed mannequin)
MOCK_IMAGE_URL = "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=512"


def set_api_key(key: str):
    """Set the fal.ai API key."""
    global _api_key
    _api_key = key


def set_mock_mode(enabled: bool = True):
    """Enable or disable mock mode for testing without API credits."""
    global _mock_mode
    _mock_mode = enabled


def generate_tryon(
    user_id: str,
    base_body_path: str,
    garment_path: str,
    garment_type: str = "upper_body",
    output_dir: str = "./output"
) -> dict:
    """
    Generate a try-on image.
    
    If no API key is set, returns a mock result with a placeholder image.
    This allows Flutter app development without needing fal.ai credits.
    
    Args:
        user_id: User identifier
        base_body_path: Path to base body image
        garment_path: Path to segmented garment
        garment_type: 'top', 'bottom', or 'dress'
        output_dir: Output directory
        
    Returns:
        Dictionary with generation results
    """
    tryon_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Mock mode: return placeholder when no API key
    if not _api_key or _mock_mode:
        return {
            "success": True,
            "tryon_image_path": None,
            "tryon_image_url": MOCK_IMAGE_URL,
            "tryon_id": f"mock_{tryon_id}",
            "processed_at": datetime.now().isoformat(),
            "error": None,
            "mock": True,  # Flag indicating this is mock data
            "message": "Mock mode: Using placeholder image. Set FAL_API_KEY for real try-on."
        }
    
    # Real mode: use fal.ai API
    service = IDMVTONService(api_key=_api_key, output_dir=output_dir)
    result = service.generate_tryon(user_id, base_body_path, garment_path, garment_type)
    return result.to_dict()


# === CLI for testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python tryon_generator.py <api_key> <user_id> <base_body_path> <garment_path> [garment_type]")
        print("Example: python tryon_generator.py your-api-key user123 ./output/user123/base_body.png ./output/user123/garment.png top")
        sys.exit(1)
    
    api_key = sys.argv[1]
    user_id = sys.argv[2]
    base_body_path = sys.argv[3]
    garment_path = sys.argv[4]
    garment_type = sys.argv[5] if len(sys.argv) > 5 else "top"
    
    import json
    
    set_api_key(api_key)
    
    print(f"Generating try-on for: {user_id}")
    print(f"  Base body: {base_body_path}")
    print(f"  Garment: {garment_path}")
    print(f"  Type: {garment_type}")
    print("-" * 40)
    
    result = generate_tryon(user_id, base_body_path, garment_path, garment_type)
    
    print(json.dumps(result, indent=2))
