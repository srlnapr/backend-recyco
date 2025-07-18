from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from app.services.prediction_service import waste_classifier

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    """
    Upload gambar dan dapatkan prediksi klasifikasi sampah
    """
    try:
        # Validasi file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File harus berupa gambar (jpg, png, jpeg)"
            )
        
        # Baca file
        contents = await file.read()
        
        # Validasi ukuran file (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Ukuran file terlalu besar (max 10MB)"
            )
        
        # Convert ke PIL Image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="File gambar tidak valid atau rusak"
            )
        
        # Debug: Check waste_classifier type
        logger.info(f"waste_classifier type: {type(waste_classifier)}")
        logger.info(f"waste_classifier attributes: {dir(waste_classifier)}")
        
        # Check if waste_classifier has predict method
        if not hasattr(waste_classifier, 'predict'):
            raise HTTPException(
                status_code=500,
                detail="Model tidak memiliki method predict"
            )
        
        # Check if predict is callable
        if not callable(getattr(waste_classifier, 'predict')):
            raise HTTPException(
                status_code=500,
                detail="Method predict tidak dapat dipanggil"
            )
        
        # Make prediction
        try:
            prediction_result = waste_classifier.predict(image)
            logger.info(f"Prediction result type: {type(prediction_result)}")
            logger.info(f"Prediction result: {prediction_result}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saat melakukan prediksi: {str(e)}"
            )
        
        # Add metadata
        result = {
            "success": True,
            "filename": file.filename,
            "file_size": len(contents),
            "image_size": image.size,
            "prediction": prediction_result,
            "message": "Prediksi berhasil"
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}"
        )

@router.get("/classes")
async def get_classes():
    """
    Dapatkan daftar kelas yang didukung
    """
    try:
        # Check if waste_classifier has classes attribute
        if not hasattr(waste_classifier, 'classes'):
            return {
                "error": "Model tidak memiliki atribut classes",
                "classes": [],
                "total_classes": 0
            }
        
        classes = waste_classifier.classes
        return {
            "classes": classes,
            "total_classes": len(classes) if classes else 0,
            "description": "Kategori sampah yang dapat diklasifikasi"
        }
    except Exception as e:
        logger.error(f"Error getting classes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error mendapatkan daftar kelas: {str(e)}"
        )

@router.get("/model-info")
async def get_model_info():
    """
    Dapatkan informasi model
    """
    try:
        # Get basic info about waste_classifier
        info = {
            "classifier_type": str(type(waste_classifier)),
            "has_predict": hasattr(waste_classifier, 'predict'),
            "has_classes": hasattr(waste_classifier, 'classes'),
            "has_model": hasattr(waste_classifier, 'model'),
            "has_device": hasattr(waste_classifier, 'device'),
        }
        
        # Try to get additional info if attributes exist
        if hasattr(waste_classifier, 'classes'):
            info["classes"] = waste_classifier.classes
        
        if hasattr(waste_classifier, 'device'):
            info["device"] = str(waste_classifier.device)
        
        if hasattr(waste_classifier, 'model'):
            info["model_status"] = "loaded" if waste_classifier.model else "not_loaded"
        
        info["input_size"] = [224, 224]  # Default assumption
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error mendapatkan informasi model: {str(e)}"
        )

@router.get("/debug")
async def debug_classifier():
    """
    Debug endpoint untuk memeriksa waste_classifier
    """
    try:
        debug_info = {
            "type": str(type(waste_classifier)),
            "is_dict": isinstance(waste_classifier, dict),
            "is_callable": callable(waste_classifier),
            "attributes": dir(waste_classifier),
            "str_representation": str(waste_classifier)[:200] + "..." if len(str(waste_classifier)) > 200 else str(waste_classifier)
        }
        
        # If it's a dict, show its keys
        if isinstance(waste_classifier, dict):
            debug_info["dict_keys"] = list(waste_classifier.keys())
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"error": str(e)}