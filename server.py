"""
TryOn Backend - API Server
FastAPI endpoints for face extraction, base body creation, garment segmentation, and try-on.
"""

import os
import shutil
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Load .env file
load_dotenv()

from face_extractor import extract_face_and_skin
from hair_extractor import extract_hair
from base_body_creator import create_base_body
from garment_segmenter import segment_garment_from_file, segment_garment_from_url
from tryon_generator import generate_tryon, set_api_key
from face_compositor import composite_face, composite_face_and_hair
from product_search import search_products




# Initialize FastAPI
app = FastAPI(
    title="TryOn Backend",
    description="Virtual Try-On Backend Services",
    version="1.0.0"
)

# CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./output"
TEMPLATES_DIR = "./assets/base_bodies"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set API key from environment
FAL_API_KEY = os.getenv("FAL_API_KEY", "")
if FAL_API_KEY:
    set_api_key(FAL_API_KEY)

# Initialize Cloudinary Storage
from cloudinary_storage import init_cloudinary
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY:
    try:
        init_cloudinary(CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)
        print(f"✅ Cloudinary Storage initialized: {CLOUDINARY_CLOUD_NAME}")
    except Exception as e:
        print(f"⚠️ Cloudinary init failed: {e}")
else:
    print("⚠️ CLOUDINARY credentials not set, uploads will be skipped")


# === Response Models ===

class FaceExtractionResponse(BaseModel):
    success: bool
    face_image_path: Optional[str] = None
    face_image_url: Optional[str] = None  # Cloudinary URL
    skin_tone_rgb: Optional[List[int]] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None


class HairExtractionResponse(BaseModel):
    success: bool
    hair_mask_path: Optional[str] = None
    hair_mask_url: Optional[str] = None  # Cloudinary URL
    hair_image_path: Optional[str] = None
    hair_image_url: Optional[str] = None  # Cloudinary URL
    processed_at: Optional[str] = None
    error: Optional[str] = None


class BaseBodyResponse(BaseModel):
    success: bool
    base_body_image_path: Optional[str] = None
    base_body_image_url: Optional[str] = None  # Cloudinary URL
    processed_at: Optional[str] = None
    error: Optional[str] = None


class GarmentSegmentationResponse(BaseModel):
    success: bool
    garment_image_path: Optional[str] = None
    garment_image_url: Optional[str] = None  # Cloudinary URL
    garment_id: Optional[str] = None
    garment_type: Optional[str] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None


class TryOnResponse(BaseModel):
    success: bool
    tryon_image_path: Optional[str] = None
    tryon_image_url: Optional[str] = None  # Cloudinary URL
    tryon_id: Optional[str] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None


class CompositeResponse(BaseModel):
    success: bool
    final_image_path: Optional[str] = None
    final_image_url: Optional[str] = None  # Cloudinary URL
    final_id: Optional[str] = None
    processed_at: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# === Request Models ===

class CreateBaseBodyRequest(BaseModel):
    user_id: str
    gender: str
    body_build: str
    height_cm: int


class SegmentGarmentUrlRequest(BaseModel):
    user_id: str
    image_url: str
    garment_id: Optional[str] = None


class TryOnRequest(BaseModel):
    user_id: str
    garment_path: str
    garment_type: str = "top"


class CompositeRequest(BaseModel):
    user_id: str
    tryon_image_path: str
    skin_tone_rgb: List[int]
    hair_image_path: Optional[str] = None  # Optional - auto-detected if not provided


class SetApiKeyRequest(BaseModel):
    api_key: str


class ProductSearchRequest(BaseModel):
    image_url: str
    max_results: int = 10
    country: str = "in"
    include_global: bool = True


class ProductSearchResponse(BaseModel):
    success: bool
    query_image: Optional[str] = None
    products: List[dict] = []
    total_found: int = 0
    search_time_ms: Optional[int] = None
    searched_at: Optional[str] = None
    error: Optional[str] = None


# === Endpoints ===

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/api/set-api-key")
async def api_set_api_key(request: SetApiKeyRequest):
    """Set the fal.ai API key."""
    set_api_key(request.api_key)
    return {"success": True, "message": "API key set"}


@app.post("/api/extract-face", response_model=FaceExtractionResponse)
async def extract_face(
    user_id: str,
    selfie: UploadFile = File(..., description="Front-facing selfie image")
):
    """Extract face and skin tone from a selfie image."""
    if not selfie.content_type or not selfie.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_extension = selfie.filename.split(".")[-1] if selfie.filename else "jpg"
    temp_filename = f"{user_id}_{uuid.uuid4().hex[:8]}.{file_extension}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(selfie.file, buffer)
        
        result = extract_face_and_skin(
            image_path=temp_path,
            user_id=user_id,
            output_dir=OUTPUT_DIR
        )
        
        return FaceExtractionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/extract-hair", response_model=HairExtractionResponse)
async def api_extract_hair(
    user_id: str,
    selfie: UploadFile = File(..., description="Front-facing selfie image")
):
    """Extract hair mask from a selfie image. Run alongside extract-face."""
    if not selfie.content_type or not selfie.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_extension = selfie.filename.split(".")[-1] if selfie.filename else "jpg"
    temp_filename = f"{user_id}_hair_{uuid.uuid4().hex[:8]}.{file_extension}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(selfie.file, buffer)
        
        result = extract_hair(
            image_path=temp_path,
            user_id=user_id,
            output_dir=OUTPUT_DIR
        )
        
        return HairExtractionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/create-base-body", response_model=BaseBodyResponse)
async def api_create_base_body(request: CreateBaseBodyRequest):
    """Create a personalized base body image from templates."""
    if request.gender not in ['male', 'female', 'other']:
        raise HTTPException(status_code=400, detail="Invalid gender")
    
    if request.body_build not in ['slim', 'regular', 'chubby', 'muscular']:
        raise HTTPException(status_code=400, detail="Invalid body_build")
    
    if not (140 <= request.height_cm <= 210):
        raise HTTPException(status_code=400, detail="height_cm must be 140-210")
    
    try:
        result = create_base_body(
            user_id=request.user_id,
            gender=request.gender,
            body_build=request.body_build,
            height_cm=request.height_cm,
            templates_dir=TEMPLATES_DIR,
            output_dir=OUTPUT_DIR
        )
        
        return BaseBodyResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segment-garment", response_model=GarmentSegmentationResponse)
async def segment_garment_upload(
    user_id: str,
    garment_id: Optional[str] = None,
    image: UploadFile = File(..., description="Outfit/garment image")
):
    """Segment garment from an uploaded image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_extension = image.filename.split(".")[-1] if image.filename else "jpg"
    temp_filename = f"{user_id}_garment_{uuid.uuid4().hex[:8]}.{file_extension}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        result = segment_garment_from_file(
            image_path=temp_path,
            user_id=user_id,
            garment_id=garment_id,
            output_dir=OUTPUT_DIR
        )
        
        return GarmentSegmentationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/segment-garment-url", response_model=GarmentSegmentationResponse)
async def segment_garment_url(request: SegmentGarmentUrlRequest):
    """Download and segment garment from a URL."""
    try:
        result = segment_garment_from_url(
            image_url=request.image_url,
            user_id=request.user_id,
            garment_id=request.garment_id,
            output_dir=OUTPUT_DIR
        )
        
        return GarmentSegmentationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tryon", response_model=TryOnResponse)
async def api_tryon(request: TryOnRequest):
    """
    Generate a try-on image using IDM-VTON.
    Uses the user's cached base body + segmented garment.
    """
    base_body_path = os.path.join(OUTPUT_DIR, request.user_id, "base_body.png")
    
    if not os.path.exists(base_body_path):
        raise HTTPException(
            status_code=400, 
            detail="Base body not found. Complete onboarding first."
        )
    
    if not os.path.exists(request.garment_path):
        raise HTTPException(
            status_code=400, 
            detail="Garment image not found."
        )
    
    try:
        result = generate_tryon(
            user_id=request.user_id,
            base_body_path=base_body_path,
            garment_path=request.garment_path,
            garment_type=request.garment_type,
            output_dir=OUTPUT_DIR
        )
        
        return TryOnResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/composite", response_model=CompositeResponse)
async def api_composite(request: CompositeRequest):
    """
    Composite user face and hair onto try-on result.
    This is the final step to create a personalized try-on image.
    """
    # Get face image path
    face_path = os.path.join(OUTPUT_DIR, f"{request.user_id}_face.png")
    
    if not os.path.exists(face_path):
        raise HTTPException(
            status_code=400,
            detail="Face image not found. Complete onboarding first."
        )
    
    if not os.path.exists(request.tryon_image_path):
        raise HTTPException(
            status_code=400,
            detail="Try-on image not found."
        )
    
    if len(request.skin_tone_rgb) != 3:
        raise HTTPException(
            status_code=400,
            detail="skin_tone_rgb must have exactly 3 values [R, G, B]"
        )
    
    # Get hair image path (auto-detect if not provided)
    hair_path = request.hair_image_path
    if not hair_path:
        hair_path = os.path.join(OUTPUT_DIR, f"{request.user_id}_hair.png")
    
    if not os.path.exists(hair_path):
        hair_path = None  # Hair is optional, continue without it
    
    try:
        result = composite_face_and_hair(
            user_id=request.user_id,
            tryon_body_path=request.tryon_image_path,
            face_path=face_path,
            hair_path=hair_path if hair_path else face_path,  # Use face as fallback
            skin_tone_rgb=request.skin_tone_rgb,
            output_dir=OUTPUT_DIR
        )
        
        return CompositeResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/face/{user_id}")
async def get_face_image(user_id: str):
    """Get the extracted face image for a user."""
    face_path = os.path.join(OUTPUT_DIR, f"{user_id}_face.png")
    
    if not os.path.exists(face_path):
        raise HTTPException(status_code=404, detail="Face image not found")
    
    return FileResponse(face_path, media_type="image/png")


@app.get("/api/base-body/{user_id}")
async def get_base_body_image(user_id: str):
    """Get the base body image for a user."""
    base_body_path = os.path.join(OUTPUT_DIR, user_id, "base_body.png")
    
    if not os.path.exists(base_body_path):
        raise HTTPException(status_code=404, detail="Base body image not found")
    
    return FileResponse(base_body_path, media_type="image/png")


@app.get("/api/garment/{user_id}/{garment_filename}")
async def get_garment_image(user_id: str, garment_filename: str):
    """Get a segmented garment image."""
    garment_path = os.path.join(OUTPUT_DIR, user_id, garment_filename)
    
    if not os.path.exists(garment_path):
        raise HTTPException(status_code=404, detail="Garment image not found")
    
    return FileResponse(garment_path, media_type="image/png")


@app.get("/api/tryon/{user_id}/{tryon_filename}")
async def get_tryon_image(user_id: str, tryon_filename: str):
    """Get a try-on result image."""
    tryon_path = os.path.join(OUTPUT_DIR, user_id, tryon_filename)
    
    if not os.path.exists(tryon_path):
        raise HTTPException(status_code=404, detail="Try-on image not found")
    
    return FileResponse(tryon_path, media_type="image/png")


# === Product Search ===

@app.post("/api/search-products", response_model=ProductSearchResponse)
async def api_search_products(request: ProductSearchRequest):
    """
    Search for products matching a garment image.
    
    Uses reverse image search to find similar products across
    e-commerce platforms with prices and purchase links.
    
    Note: Requires SERPAPI_KEY environment variable to be set.
    """
    try:
        result = search_products(
            image_url=request.image_url,
            max_results=request.max_results,
            country=request.country,
            include_global=request.include_global
        )
        return ProductSearchResponse(**result)
        
    except Exception as e:
        return ProductSearchResponse(
            success=False,
            error=str(e)
        )


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
