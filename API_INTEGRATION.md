# FastAPI Integration Guide for Multimodal Input

## Overview
The Math Tutoring Agent now supports **image inputs** in addition to text. Students can send photos of their work, handwritten problems, or diagrams. The agent automatically extracts text using OCR (Optical Character Recognition) and processes it.

---

## How It Works

### Architecture
```
Mobile App → FastAPI → Save Image to Temp File → Send File Path → Agent Graph → OCR Processing → Text Extraction → Tutoring Flow
```

The agent's graph automatically detects and processes:
1. **Regular text** - Passes through unchanged
2. **Image file paths** - Runs OCR to extract text
3. **Base64 images** - Decodes and runs OCR (used by LangGraph Studio)

---

## FastAPI Endpoint Implementation

### Update `/session/continue` Endpoint

Add support for multipart file uploads to the existing endpoint:

```python
from fastapi import FastAPI, File, UploadFile, Form
import uuid
import os
from pathlib import Path

# Create temp directory for uploaded images
UPLOAD_DIR = Path("./temp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/session/continue")
async def continue_session(
    thread_id: str = Form(...),
    user_message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    clicked_autosuggestion: Optional[bool] = Form(False),
    student_level: Optional[str] = Form(None)
):
    """
    Continue session with text OR image input.
    
    Args:
        thread_id: Session thread ID
        user_message: Text message from student (optional if image provided)
        image: Image file uploaded by student (optional if user_message provided)
        clicked_autosuggestion: Whether user clicked a suggestion
        student_level: Student ability level
    """
    
    # STEP 1: Determine input type and prepare message content
    if image:
        # Save uploaded image to temporary file
        file_extension = os.path.splitext(image.filename)[1] or ".jpg"
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = UPLOAD_DIR / temp_filename
        
        # Write image bytes to file
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Send file path to agent (agent will run OCR automatically)
        message_content = str(temp_path)
        
        print(f"📸 Image uploaded: {image.filename} → {temp_path}")
        
    elif user_message:
        # Regular text input
        message_content = user_message
    else:
        raise HTTPException(
            status_code=400, 
            detail="Either user_message or image must be provided"
        )
    
    # STEP 2: Create HumanMessage with content (text or file path)
    user_msg = HumanMessage(content=message_content)
    
    # STEP 3: Invoke graph (OCR happens automatically inside graph wrapper)
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(
        Command(resume={"messages": [user_msg]}),
        config=config
    )
    
    # STEP 4: Clean up temp file (optional - you may want to keep for debugging)
    # if image and temp_path.exists():
    #     os.unlink(temp_path)
    
    # Rest of your existing response building...
    # (extract agent message, metadata, etc.)
```

---

## Mobile App Implementation

### Example: React Native

```javascript
const sendImageToAgent = async (photoUri, threadId) => {
  const formData = new FormData();
  formData.append('thread_id', threadId);
  formData.append('image', {
    uri: photoUri,
    type: 'image/jpeg',
    name: 'student_work.jpg'
  });

  const response = await fetch('https://your-api.com/session/continue', {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });

  return await response.json();
};
```

### Example: Flutter

```dart
Future<void> sendImageToAgent(String imagePath, String threadId) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('https://your-api.com/session/continue')
  );
  
  request.fields['thread_id'] = threadId;
  request.files.add(
    await http.MultipartFile.fromPath('image', imagePath)
  );
  
  var response = await request.send();
  var responseBody = await response.stream.bytesToString();
  
  return jsonDecode(responseBody);
}
```

---

## Request Examples

### Text Input (Existing Behavior)
```bash
curl -X POST "http://localhost:8000/session/continue" \
  -F "thread_id=abc-123" \
  -F "user_message=I want to add 1/2 and 1/3"
```

### Image Input (NEW)
```bash
curl -X POST "http://localhost:8000/session/continue" \
  -F "thread_id=abc-123" \
  -F "image=@student_work.jpg"
```

### Both Text and Image
```bash
# Note: If both provided, image takes precedence
curl -X POST "http://localhost:8000/session/continue" \
  -F "thread_id=abc-123" \
  -F "user_message=Here is my work" \
  -F "image=@photo.png"
```

---

## Important Notes

### 1. **No Changes to Response Format**
The API response remains exactly the same. The agent processes images internally and responds with text as usual.

### 2. **Temp File Management**
- Images are saved to `./temp/uploads/` directory
- Files are named with UUIDs to avoid conflicts
- Consider implementing periodic cleanup of old temp files
- Optional: Delete files after processing (commented out in example)

### 3. **Error Handling**
If OCR fails, the agent will respond with:
```
"[Error: Could not read image. Please type your response instead.]"
```

The mobile app should handle this gracefully and prompt the user to type their response.

### 4. **File Size Limits**
Add file size validation:
```python
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

if image and image.size > MAX_IMAGE_SIZE:
    raise HTTPException(
        status_code=413,
        detail="Image too large. Maximum size is 10MB."
    )
```

### 5. **Supported Image Formats**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- WebP (.webp)
- GIF (.gif)

---

## Testing

### Test with LangGraph Studio
Images work automatically in Studio - just upload an image and the agent will process it.

### Test with Local File
```python
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="./test_image.jpg")]
}, config)
```

### Test with FastAPI
Use Postman or curl with multipart form data.

---

## Cleanup Script (Optional)

To periodically clean old temp files:

```python
import os
import time
from pathlib import Path

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Delete files older than max_age_hours."""
    now = time.time()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            age_hours = (now - file_path.stat().st_mtime) / 3600
            if age_hours > max_age_hours:
                os.unlink(file_path)
                print(f"Deleted old file: {file_path}")

# Run periodically (e.g., in a background task)
cleanup_old_files(UPLOAD_DIR, max_age_hours=24)
```

---

## Summary

**For API Developers:**
1. Accept `UploadFile` parameter named `image` in `/session/continue`
2. Save uploaded file to temp directory
3. Pass file path as `HumanMessage` content
4. Agent automatically detects and processes image
5. No changes needed to response handling

**Agent handles everything else automatically!** 🚀
