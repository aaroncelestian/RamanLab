from PIL import Image, ImageDraw, ImageFont
import os

def create_gradient_background(size, start_color, end_color):
    """Create a gradient background from start_color to end_color."""
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    for y in range(size[1]):
        # Calculate gradient color for this row
        r = int(start_color[0] + (end_color[0] - start_color[0]) * y / size[1])
        g = int(start_color[1] + (end_color[1] - start_color[1]) * y / size[1])
        b = int(start_color[2] + (end_color[2] - start_color[2]) * y / size[1])
        draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
    
    return image

def create_icon(size):
    """Create the RamanLab icon at the specified size."""
    # Colors
    deep_blue = (5, 10, 20)  # Almost black with slight blue tint
    white = (255, 255, 255)  # Pure white
    gradient_end = (10, 15, 30)  # Slightly lighter almost-black for gradient
    
    # Create base image with gradient
    image = create_gradient_background(size, deep_blue, gradient_end)
    draw = ImageDraw.Draw(image)
    
    # Calculate font size (reduced by 20% from previous 80%)
    font_size = int(min(size) * 0.64)  # 0.8 * 0.8 = 0.64
    
    try:
        # Try to use a modern sans-serif font
        font = ImageFont.truetype("Arial", font_size)
    except:
        # Fallback to default font if Arial is not available
        font = ImageFont.load_default()
    
    # Calculate text position to align with left and bottom
    text = "RL"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position text to perfectly align with left and bottom edges
    # Move the entire text group slightly left
    x = -int(size[0] * 0.05)  # Move 5% of the width to the left (reduced from 10%)
    y = size[1] - text_height - text_bbox[1]  # Keep bottom alignment
    
    # Draw the text with white color
    draw.text((x, y), text, font=font, fill=white)
    
    return image

def main():
    # Create icons in different sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    for size in sizes:
        icon = create_icon((size, size))
        
        # Save as PNG
        icon.save(f"RamanLab_{size}.png")
        
        # For macOS, also save as ICNS
        if size == 1024:  # Only create ICNS from the largest size
            icon.save("RamanLab.icns")
    
    print("Icons created successfully!")

if __name__ == "__main__":
    main() 