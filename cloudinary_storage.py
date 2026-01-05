"""
TryOn Backend - Cloudinary Storage Utility
Handles uploading files to Cloudinary and returning public URLs.

Cloudinary Free Tier: 25GB storage, 25K transformations/month
"""

import os
import cloudinary
import cloudinary.uploader
from typing import Optional
from datetime import datetime


class CloudinaryStorage:
    """
    Cloudinary Storage utility for uploading VTON images.
    
    Storage Structure (using folders):
    users/{userId}/vton/
    ├── face
    ├── hair
    ├── hair_mask
    ├── base_body
    ├── garments/{garmentId}
    ├── tryons/{tryonId}
    └── finals/{finalId}
    """
    
    _initialized = False
    
    @classmethod
    def initialize(cls, cloud_name: str = None, api_key: str = None, api_secret: str = None):
        """
        Initialize Cloudinary SDK.
        
        Args:
            cloud_name: Cloudinary cloud name
            api_key: Cloudinary API key
            api_secret: Cloudinary API secret
        """
        if cls._initialized:
            return
        
        cloudinary.config(
            cloud_name=cloud_name or os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=api_key or os.getenv('CLOUDINARY_API_KEY'),
            api_secret=api_secret or os.getenv('CLOUDINARY_API_SECRET'),
            secure=True
        )
        cls._initialized = True
    
    @classmethod
    def upload_file(
        cls,
        local_path: str,
        public_id: str,
        folder: str = None
    ) -> Optional[str]:
        """
        Upload a file to Cloudinary and return public URL.
        
        Args:
            local_path: Path to local file
            public_id: Unique identifier for the image
            folder: Optional folder path in Cloudinary
            
        Returns:
            Public URL of uploaded file, or None if failed
        """
        if not cls._initialized:
            print("Warning: Cloudinary not initialized, skipping upload")
            return None
        
        try:
            result = cloudinary.uploader.upload(
                local_path,
                public_id=public_id,
                folder=folder,
                resource_type="image",
                overwrite=True
            )
            return result.get('secure_url')
        except Exception as e:
            print(f"Cloudinary upload error: {e}")
            return None
    
    @classmethod
    def upload_face(cls, user_id: str, local_path: str) -> Optional[str]:
        """Upload face image."""
        return cls.upload_file(
            local_path,
            public_id="face",
            folder=f"users/{user_id}/vton"
        )
    
    @classmethod
    def upload_hair(cls, user_id: str, local_path: str) -> Optional[str]:
        """Upload hair image."""
        return cls.upload_file(
            local_path,
            public_id="hair",
            folder=f"users/{user_id}/vton"
        )
    
    @classmethod
    def upload_hair_mask(cls, user_id: str, local_path: str) -> Optional[str]:
        """Upload hair mask."""
        return cls.upload_file(
            local_path,
            public_id="hair_mask",
            folder=f"users/{user_id}/vton"
        )
    
    @classmethod
    def upload_base_body(cls, user_id: str, local_path: str) -> Optional[str]:
        """Upload base body image."""
        return cls.upload_file(
            local_path,
            public_id="base_body",
            folder=f"users/{user_id}/vton"
        )
    
    @classmethod
    def upload_garment(cls, user_id: str, garment_id: str, local_path: str) -> Optional[str]:
        """Upload garment image."""
        return cls.upload_file(
            local_path,
            public_id=garment_id,
            folder=f"users/{user_id}/vton/garments"
        )
    
    @classmethod
    def upload_tryon(cls, user_id: str, tryon_id: str, local_path: str) -> Optional[str]:
        """Upload try-on result image."""
        return cls.upload_file(
            local_path,
            public_id=tryon_id,
            folder=f"users/{user_id}/vton/tryons"
        )
    
    @classmethod
    def upload_final(cls, user_id: str, final_id: str, local_path: str) -> Optional[str]:
        """Upload final composite image."""
        return cls.upload_file(
            local_path,
            public_id=final_id,
            folder=f"users/{user_id}/vton/finals"
        )
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique ID for files."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# Convenience function for initialization
def init_cloudinary(cloud_name: str = None, api_key: str = None, api_secret: str = None):
    """Initialize Cloudinary Storage."""
    CloudinaryStorage.initialize(cloud_name, api_key, api_secret)
