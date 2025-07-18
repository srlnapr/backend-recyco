from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io

class ImageProcessor:
    @staticmethod
    def enhance_image(image: Image.Image, enhance_factor: float = 1.2) -> Image.Image:
        """
        Tingkatkan kualitas gambar
        """
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhance_factor)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(enhance_factor)
            
            return image
        except Exception as e:
            return image
    
    @staticmethod
    def remove_noise(image: Image.Image) -> Image.Image:
        """
        Hilangkan noise dari gambar
        """
        try:
            # Apply mild blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            return image
        except Exception as e:
            return image
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: tuple = (224, 224)) -> Image.Image:
        """
        Resize gambar dengan mempertahankan aspek rasio
        """
        try:
            # Calculate aspect ratio
            aspect_ratio = image.width / image.height
            
            if aspect_ratio > 1:  # Landscape
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect_ratio)
            else:  # Portrait
                new_height = target_size[1]
                new_width = int(target_size[1] * aspect_ratio)
            
            # Resize
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', target_size, (255, 255, 255))
            
            # Center the image
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            
            new_image.paste(resized, (x_offset, y_offset))
            
            return new_image
        except Exception as e:
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def normalize_image(image: Image.Image) -> Image.Image:
        """
        Normalisasi gambar
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize to 0-255 range
            img_array = np.clip(img_array, 0, 255)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array.astype(np.uint8))
        except Exception as e:
            return image
    
    @staticmethod
    def preprocess_for_model(image: Image.Image) -> Image.Image:
        """
        Preprocessing lengkap untuk model
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image quality
            image = ImageProcessor.enhance_image(image)
            
            # Remove noise
            image = ImageProcessor.remove_noise(image)
            
            # Resize to model input size
            image = ImageProcessor.resize_image(image, (224, 224))
            
            # Normalize
            image = ImageProcessor.normalize_image(image)
            
            return image
        except Exception as e:
            # Fallback to simple resize
            return image.resize((224, 224), Image.Resampling.LANCZOS)
    
    @staticmethod
    def validate_image(image_bytes: bytes) -> bool:
        """
        Validasi apakah file adalah gambar yang valid
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """
        Dapatkan informasi gambar
        """
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }