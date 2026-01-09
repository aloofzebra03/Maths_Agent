#!/usr/bin/env python3
"""
Script to create placeholder character images for the lip-sync animation.
Run this if you don't have myphoto.png and myphoto2.png in your directory.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_character_image(name, color, filename):
    """Create a simple character avatar"""
    # Create a 300x300 image with a circular background
    size = (300, 300)
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw circular background
    margin = 20
    circle_bbox = [margin, margin, size[0]-margin, size[1]-margin]
    draw.ellipse(circle_bbox, fill=color, outline='#333', width=5)
    
    # Draw simple face features
    # Eyes
    eye_y = size[1] // 3
    eye_size = 25
    left_eye = [size[0]//3 - eye_size//2, eye_y, size[0]//3 + eye_size//2, eye_y + eye_size]
    right_eye = [2*size[0]//3 - eye_size//2, eye_y, 2*size[0]//3 + eye_size//2, eye_y + eye_size]
    
    draw.ellipse(left_eye, fill='white', outline='black', width=2)
    draw.ellipse(right_eye, fill='white', outline='black', width=2)
    
    # Eye pupils
    pupil_size = 8
    left_pupil = [size[0]//3 - pupil_size//2, eye_y + eye_size//2 - pupil_size//2, 
                  size[0]//3 + pupil_size//2, eye_y + eye_size//2 + pupil_size//2]
    right_pupil = [2*size[0]//3 - pupil_size//2, eye_y + eye_size//2 - pupil_size//2,
                   2*size[0]//3 + pupil_size//2, eye_y + eye_size//2 + pupil_size//2]
    
    draw.ellipse(left_pupil, fill='black')
    draw.ellipse(right_pupil, fill='black')
    
    # Nose (simple line)
    nose_x = size[0] // 2
    nose_start_y = eye_y + eye_size + 20
    nose_end_y = nose_start_y + 20
    draw.line([(nose_x, nose_start_y), (nose_x, nose_end_y)], fill='black', width=2)
    
    # Mouth will be animated, so just draw a neutral expression
    mouth_y = nose_end_y + 30
    mouth_width = 40
    mouth_rect = [nose_x - mouth_width//2, mouth_y, nose_x + mouth_width//2, mouth_y + 10]
    draw.ellipse(mouth_rect, fill='#cc3333', outline='black', width=2)
    
    # Add character label
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (size[0] - text_width) // 2
    text_y = size[1] - 50
    
    draw.text((text_x, text_y), name, fill='black', font=font)
    
    # Save the image
    image.save(filename)
    print(f"Created {filename}")

def main():
    """Create both character images"""
    # Boy character (blue theme)
    create_character_image("Boy", "#87CEEB", "myphoto2.png")
    
    # Girl character (pink theme) 
    create_character_image("Girl", "#FFB6C1", "myphoto.png")
    
    print("Character images created successfully!")
    print("You can replace these with your own images if desired.")

if __name__ == "__main__":
    main()
