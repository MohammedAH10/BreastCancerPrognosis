import os
import uuid
from werkzeug.utils import secure_filename

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_image(image_file, upload_folder):
    """Save uploaded image and return filename"""
    try:
        # Generate unique filename
        file_extension = os.path.splitext(image_file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(upload_folder, filename)
        
        # Save the file
        image_file.save(filepath)
        
        return filename
        
    except Exception as e:
        raise Exception(f"Error saving image: {str(e)}")