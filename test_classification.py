from models import EyeSeverityDetector
import numpy as np
from PIL import Image

detector = EyeSeverityDetector()
print('Detector initialized')

# Test with different types of images
# Test 1: Low pathology image (should be Normal)
low_pathology = np.zeros((224, 224, 3), dtype=np.uint8)
low_pathology[:, :, 0] = 120  # Moderate red
low_pathology[:, :, 1] = 140  # Moderate green
low_pathology[:, :, 2] = 130  # Moderate blue
pil_img1 = Image.fromarray(low_pathology)
result1 = detector.classify_severity(pil_img1)
print('Low pathology result:', result1['severity'], f'({result1["score"]:.2f})')

# Test 2: High pathology image (should be Severe)
high_pathology = np.zeros((224, 224, 3), dtype=np.uint8)
high_pathology[:, :, 0] = 200  # High red
high_pathology[:, :, 1] = 100  # Low green
high_pathology[:, :, 2] = 100  # Low blue
# Add some dark spots
high_pathology[50:80, 50:80] = [50, 50, 50]
pil_img2 = Image.fromarray(high_pathology)
result2 = detector.classify_severity(pil_img2)
print('High pathology result:', result2['severity'], f'({result2["score"]:.2f})')

# Test 3: Moderate pathology
moderate_pathology = np.zeros((224, 224, 3), dtype=np.uint8)
moderate_pathology[:, :, 0] = 160  # Higher red
moderate_pathology[:, :, 1] = 120  # Lower green
moderate_pathology[:, :, 2] = 120  # Lower blue
pil_img3 = Image.fromarray(moderate_pathology)
result3 = detector.classify_severity(pil_img3)
print('Moderate pathology result:', result3['severity'], f'({result3["score"]:.2f})')