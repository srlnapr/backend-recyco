import requests
import json
from pathlib import Path

# Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_get_classes():
    """Test get classes endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/classes")
        print(f"Get Classes Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Get classes failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/model-info")
        print(f"Model Info Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_prediction(image_path):
    """Test prediction endpoint"""
    try:
        if not Path(image_path).exists():
            print(f"Image file not found: {image_path}")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
        
        print(f"Prediction Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction Result:")
            print(f"  - Predicted Class: {result['prediction']['predicted_class']}")
            print(f"  - Confidence: {result['prediction']['confidence']:.4f}")
            print(f"  - All Probabilities:")
            for class_name, prob in result['prediction']['all_probabilities'].items():
                print(f"    - {class_name}: {prob:.4f}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

def main():
    print("🧪 Testing Waste Classification API")
    print("=" * 50)
    
    # Test endpoints
    tests = [
        ("Health Check", test_health_check),
        ("Get Classes", test_get_classes),
        ("Model Info", test_model_info),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        print(f"✅ PASSED" if result else "❌ FAILED")
        print("-" * 30)
    
    # Test prediction if image is available
    print(f"\n📸 Testing Prediction:")
    print("Put an image file in the same directory as this script")
    print("and update the image_path variable below")
    
    # Update this path to your test image
    image_path = "contoh_sampah.jpg"
    
    if Path(image_path).exists():
        result = test_prediction(image_path)
        print(f"✅ PASSED" if result else "❌ FAILED")
    else:
        print(f"⚠️  SKIPPED - No test image found at {image_path}")

if __name__ == "__main__":
    main()