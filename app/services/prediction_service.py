import torch
import torch.nn as nn
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import os
import logging
import sys
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WasteClassifier:
    def __init__(self):
        self.model = None
        self.classes = ['plastic', 'organic', 'paper', 'glass', 'metal', 'cardboard']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model with proper device mapping"""
        try:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'models', 'waste_classifier_6cat_production.pkl'),
                os.path.join(os.path.dirname(__file__), 'models', 'waste_classifier_6cat_production.pkl'),
                os.path.join('app', 'models', 'waste_classifier_6cat_production.pkl'),
                os.path.join('models', 'waste_classifier_6cat_production.pkl'),
                'waste_classifier_6cat_production.pkl'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                logger.error("Model file not found in any expected location")
                logger.info("Please place 'waste_classifier_6cat_production.pkl' in the app/models/ directory")
                # Create model architecture as fallback
                self.create_model_architecture()
                return
            
            logger.info(f"Loading model from: {model_path}")
            
            # Method 1: Try loading with torch.load
            try:
                # Load the model with proper device mapping
                with open(model_path, 'rb') as f:
                    # Set the default tensor type to CPU to avoid CUDA issues
                    torch.set_default_tensor_type('torch.FloatTensor')
                    
                    # Load the model (removed the extra comma)
                    loaded_data = torch.load(f, map_location=self.device)
                    
                    # Handle different formats
                    if isinstance(loaded_data, dict):
                        if 'model' in loaded_data:
                            self.model = loaded_data['model']
                        elif 'state_dict' in loaded_data:
                            # If it's a state dict, we need to create the architecture first
                            self.create_model_architecture()
                            self.model.load_state_dict(loaded_data['state_dict'])
                        else:
                            # Assume the dict itself is the model
                            self.model = loaded_data
                    else:
                        # Direct model object
                        self.model = loaded_data
                    
                    # Ensure model is on correct device and in eval mode
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(self.device)
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    
                    self.model_loaded = True
                    logger.info(f"Model loaded successfully on {self.device}")
                    return
                    
            except Exception as e:
                logger.warning(f"torch.load failed: {e}")
                
            # Method 2: Try with pickle.load
            try:
                with open(model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                    # Handle different formats
                    if isinstance(loaded_data, dict):
                        if 'model' in loaded_data:
                            self.model = loaded_data['model']
                        else:
                            self.model = loaded_data
                    else:
                        self.model = loaded_data
                    
                    # Move model to appropriate device
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(self.device)
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    
                    self.model_loaded = True
                    logger.info(f"Model loaded successfully with pickle on {self.device}")
                    return
                    
            except Exception as e:
                logger.warning(f"pickle.load also failed: {e}")
            
            # If all methods fail, create fresh architecture
            logger.error("All loading methods failed, creating fresh model architecture")
            self.create_model_architecture()
            
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            self.create_model_architecture()
    
    def create_model_architecture(self):
        """Create model architecture as fallback"""
        try:
            logger.warning("Creating model architecture without pre-trained weights")
            
            # Create EfficientNet-B2 model
            base_model = EfficientNet.from_pretrained('efficientnet-b2')
            
            # Modify classifier
            num_features = base_model._fc.in_features
            base_model._fc = nn.Linear(num_features, len(self.classes))
            
            self.model = base_model
            self.model.to(self.device)
            self.model.eval()
            
            # Mark as loaded but warn about accuracy
            self.model_loaded = True
            logger.warning("Using fresh model architecture - predictions may not be accurate")
            
        except Exception as e:
            logger.error(f"Error creating model architecture: {str(e)}")
            self.model_loaded = False
            raise
    
    def preprocess_image(self, image: Image.Image):
        """Preprocess image for prediction"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: Image.Image):
        """Make prediction on image"""
        try:
            # Check if model is loaded
            if not self.model_loaded or self.model is None:
                raise ValueError("Model not loaded properly")
            
            # Additional check to ensure model is callable
            if not hasattr(self.model, '__call__'):
                raise ValueError("Model is not callable - it might be a dictionary instead of a model object")
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            # Get class name
            predicted_class = self.classes[predicted_class_idx]
            
            # Create result
            result = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "all_probabilities": {
                    self.classes[i]: float(probabilities[0][i].item())
                    for i in range(len(self.classes))
                }
            }
            
            logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.error(f"Model type: {type(self.model)}")
            logger.error(f"Model loaded: {self.model_loaded}")
            raise

# Global instance
try:
    waste_classifier = WasteClassifier()
    logger.info("WasteClassifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize WasteClassifier: {str(e)}")
    # Create a dummy classifier to prevent import errors
    class DummyClassifier:
        def __init__(self):
            self.classes = ['plastic', 'organic', 'paper', 'glass', 'metal', 'cardboard']
            self.device = torch.device('cpu')
            self.model = None
            self.model_loaded = False
        
        def predict(self, image):
            raise ValueError("Model failed to load properly. Please check the model file.")
    
    waste_classifier = DummyClassifier()