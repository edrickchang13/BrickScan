# Local Inventory API Client Usage Guide

This guide shows how to integrate with the local inventory scanning API from a mobile or web frontend.

## Quick Start

### 1. Scan a Brick

```python
import requests
import base64

# Read image from file or camera
with open("brick_photo.jpg", "rb") as f:
    image_bytes = f.read()

# Encode as base64
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# Send scan request
response = requests.post(
    "http://localhost:8000/api/local-inventory/scan",
    json={
        "image_base64": image_base64,
        "session_id": None  # optional session grouping
    }
)

result = response.json()
print(f"Status: {result['status']}")  # 'known' or 'uncertain'
print(f"Top prediction: {result['primary_prediction']['part_num']}")
print(f"Confidence: {result['primary_prediction']['confidence']:.1%}")
```

### 2. Handle Different Statuses

**If status is "known" (≥80% confidence):**
```python
if result["status"] == "known":
    # User can quick-add or review first
    prediction = result["primary_prediction"]
    
    # Add to inventory
    response = requests.post(
        "http://localhost:8000/api/local-inventory/inventory/add",
        json={
            "part_num": prediction["part_num"],
            "color_id": prediction["color_id"],
            "color_name": prediction["color_name"],
            "quantity": 1
        }
    )
    added_part = response.json()
    print(f"Added: {added_part['part_num']} x{added_part['quantity']}")
```

**If status is "uncertain" (<80% confidence):**
```python
if result["status"] == "uncertain":
    # Show user the top-3 predictions to choose from
    predictions = result["predictions"]
    
    print("Is this part one of these?")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['part_num']} "
              f"({pred.get('color_name', 'Unknown')} color) "
              f"{pred['confidence']:.1%}")
    
    # User picks option (e.g., option 2)
    chosen = predictions[1]
    
    # Add with user confirmation
    response = requests.post(
        "http://localhost:8000/api/local-inventory/inventory/add",
        json={
            "part_num": chosen["part_num"],
            "color_id": chosen["color_id"],
            "color_name": chosen["color_name"],
            "quantity": 1
        }
    )
```

### 3. View Inventory

```python
response = requests.get("http://localhost:8000/api/local-inventory/inventory")
parts = response.json()

print(f"You own {len(parts)} different parts:")
for part in parts:
    status = "✓" if part["user_confirmed"] else "?"
    print(f"{status} {part['part_num']:8} "
          f"({part['color_name']:15}) x{part['quantity']}")
```

### 4. Get Statistics

```python
response = requests.get("http://localhost:8000/api/local-inventory/inventory/stats")
stats = response.json()

print(f"Total unique parts: {stats['total_parts']}")
print(f"Total pieces: {stats['total_quantity']}")
print(f"Confirmed: {stats['user_confirmed']}")
print(f"Uncertain: {stats['uncertain_parts']}")
```

### 5. Correct a Misprediction

```python
# If user realizes a part is wrong:
inventory_id = "some-uuid"  # from inventory listing

response = requests.post(
    f"http://localhost:8000/api/local-inventory/inventory/{inventory_id}/correct",
    json={
        "correct_part_num": "3002",  # The actual correct part
        "correct_color_id": 1,
        "correct_color_name": "White"
    }
)
corrected = response.json()
print(f"Corrected to: {corrected['part_num']}")
```

### 6. Export Inventory

```python
response = requests.get("http://localhost:8000/api/local-inventory/inventory/export")
csv_data = response.text

# Save to file
with open("my_inventory.csv", "w") as f:
    f.write(csv_data)

# Or use directly in pandas
import pandas as pd
df = pd.read_csv(io.StringIO(response.text))
print(df)
```

## Session Management

### Start a Scanning Session

```python
response = requests.post(
    "http://localhost:8000/api/local-inventory/scan-session/start",
    json={
        "set_name": "Technic 42145 - April 2024"
    }
)
session = response.json()
session_id = session["id"]

# Now scan bricks with this session_id
# (not currently used in routing, but saved for future grouping)
```

### List Sessions

```python
response = requests.get(
    "http://localhost:8000/api/local-inventory/scan-session"
)
sessions = response.json()

for s in sessions:
    status = "✓ Done" if s["completed"] else "In Progress"
    print(f"{s['set_name']:30} {status}")
```

### Complete a Session

```python
response = requests.post(
    f"http://localhost:8000/api/local-inventory/scan-session/{session_id}/complete"
)
print(f"Completed session: {response.json()['set_name']}")
```

## Mobile Implementation (React Native / Flutter)

### React Native Example

```javascript
// Capture image from camera
const image = await launchCamera();
const imageBase64 = await RNFS.readFile(image.uri, 'base64');

// Send to API
const response = await fetch('http://localhost:8000/api/local-inventory/scan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_base64: imageBase64,
    session_id: null
  })
});

const result = await response.json();

// Handle response
if (result.status === 'known') {
  // Quick-add button
  showQuickAddModal(result.primary_prediction);
} else {
  // Show picker for top-3
  showPartPicker(result.predictions);
}
```

### Flutter Example

```dart
// Capture and encode image
final pickedFile = await _picker.pickImage(source: ImageSource.camera);
final bytes = await pickedFile.readAsBytes();
final base64Image = base64Encode(bytes);

// Send to API
final response = await http.post(
  Uri.parse('http://localhost:8000/api/local-inventory/scan'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'image_base64': base64Image,
    'session_id': null
  })
);

final result = jsonDecode(response.body);

// Handle result
if (result['status'] == 'known') {
  _showQuickAdd(result['primary_prediction']);
} else {
  _showPartPicker(result['predictions']);
}
```

## Error Handling

### Network Errors

```python
import requests
from requests.exceptions import ConnectionError, Timeout

try:
    response = requests.post(
        "http://localhost:8000/api/local-inventory/scan",
        json={"image_base64": image_base64},
        timeout=5
    )
    response.raise_for_status()  # Raise for 4xx/5xx
except ConnectionError:
    print("API server not reachable")
except Timeout:
    print("Request timed out")
except requests.HTTPError as e:
    print(f"API error: {e.response.status_code} {e.response.text}")
```

### Image Validation Errors

```python
if response.status_code == 400:
    error = response.json()
    print(f"Image error: {error['detail']}")
    # Possible errors:
    # - "Invalid base64 encoding"
    # - "Image too large"
    # - "Invalid or corrupted image"
```

### Not Found Errors

```python
if response.status_code == 404:
    error = response.json()
    print(f"Not found: {error['detail']}")
    # Possible errors:
    # - "Inventory part not found"
    # - "Scan session not found"
```

## Batch Operations

### Scan Multiple Parts

```python
import time

parts_to_add = []

for i in range(10):
    # Scan a brick
    response = requests.post(
        "http://localhost:8000/api/local-inventory/scan",
        json={"image_base64": image_base64}
    )
    result = response.json()
    
    if result["status"] == "known":
        # Auto-add
        part = result["primary_prediction"]
        requests.post(
            "http://localhost:8000/api/local-inventory/inventory/add",
            json={
                "part_num": part["part_num"],
                "color_id": part["color_id"],
                "color_name": part["color_name"],
                "quantity": 1
            }
        )
    else:
        # User picks from top-3
        parts_to_add.append(result)
    
    # Small delay to avoid overwhelming API
    time.sleep(0.5)

print(f"Added {len(parts_to_add)} parts to review")
```

### Bulk Update Quantities

```python
# Get all parts
response = requests.get("http://localhost:8000/api/local-inventory/inventory")
parts = response.json()

# Update quantities (e.g., double all counts)
for part in parts:
    requests.put(
        f"http://localhost:8000/api/local-inventory/inventory/{part['id']}",
        json={"quantity": part["quantity"] * 2}
    )
```

## Performance Tips

1. **Compress images**: Send JPEG instead of PNG to reduce size
   ```python
   img = Image.open("photo.jpg")
   img.thumbnail((800, 800))  # Reduce size before scanning
   ```

2. **Batch requests**: Don't wait for response before scanning next brick
   ```python
   # Concurrent scanning
   import concurrent.futures
   
   with concurrent.futures.ThreadPoolExecutor() as executor:
       futures = [executor.submit(scan_image, img) for img in images]
       results = [f.result() for f in futures]
   ```

3. **Cache colors**: Store color lookup locally
   ```python
   color_cache = {}  # {color_id: color_name}
   
   def get_color_name(color_id):
       if color_id not in color_cache:
           # Fetch from API if needed
           pass
       return color_cache.get(color_id, "Unknown")
   ```

4. **Lazy load inventory**: Don't fetch all parts at once
   ```python
   # Pagination (future enhancement)
   response = requests.get(
       "http://localhost:8000/api/local-inventory/inventory",
       params={"limit": 50, "offset": 0}
   )
   ```

## Testing

### Mock API for Testing

```python
from unittest.mock import Mock, patch

@patch('requests.post')
def test_scan_known(mock_post):
    mock_post.return_value.json.return_value = {
        "status": "known",
        "predictions": [
            {
                "part_num": "3001",
                "confidence": 0.92,
                "color_id": 1,
                "color_name": "White"
            }
        ],
        "primary_prediction": {...}
    }
    
    # Test code
    result = scan_brick(image_base64)
    assert result["status"] == "known"
```

### Test Data

Small test images can be generated:
```python
from PIL import Image
import base64
import io

# Create a solid color image for testing
img = Image.new("RGB", (224, 224), color="white")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
test_base64 = base64.b64encode(buffer.getvalue()).decode()
```
