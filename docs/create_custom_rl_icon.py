from PIL import Image, ImageDraw, ImageFont
import os

def create_custom_rl_icon():
    """
    Create a custom RamanLab icon with:
    - R just barely touching the left side of the icon
    - L just barely merging with the R
    - Both letters at the bottom part of the visible area
    - macOS-style rounded corners and padding
    """
    
    # Start with the best available RamanLab icon
    base_img = None
    if os.path.exists("RamanLab_mac_1024.png"):
        base_img = Image.open("RamanLab_mac_1024.png").convert("RGBA")
        print("Using RamanLab_mac_1024.png as base")
    elif os.path.exists("RamanLab_mac_512.png"):
        base_img = Image.open("RamanLab_mac_512.png").convert("RGBA")
        print("Using RamanLab_mac_512.png as base")
    elif os.path.exists("RamanLab_1024.png"):
        base_img = Image.open("RamanLab_1024.png").convert("RGBA")
        print("Using RamanLab_1024.png as base")
    elif os.path.exists("RamanLab_512.png"):
        base_img = Image.open("RamanLab_512.png").convert("RGBA")
        print("Using RamanLab_512.png as base")
    else:
        print("No suitable RamanLab PNG found.")
        # List available files for debugging
        png_files = [f for f in os.listdir('.') if f.startswith('RamanLab') and f.endswith('.png')]
        print(f"Available RamanLab PNG files: {png_files}")
        return
    
    # Work with 1024x1024 for high quality
    final_size = 1024
    base_img = base_img.resize((final_size, final_size), Image.LANCZOS)
    
    # Create a new image to work with
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to use a bold system font, fallback to default if not available
    try:
        # Try different font sizes to find the right fit
        font_size = 200  # Start with a large size
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Get text dimensions for positioning
    text = "RL"
    
    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position calculations:
    # X: R should just barely touch the left side (small margin)
    x_position = 10  # Small margin from left edge
    
    # Y: Bottom part of the visible area
    # Assuming the "black box" (content area) takes up about 80% of the icon
    content_area_height = int(final_size * 0.8)
    content_start_y = int(final_size * 0.1)  # 10% from top
    
    # Position text at bottom of content area
    y_position = content_start_y + content_area_height - text_height - 20  # 20px margin from bottom
    
    # Draw the text in white (assuming dark background)
    draw.text((x_position, y_position), text, font=font, fill=(255, 255, 255, 255))
    
    # Now let's fine-tune the R and L positioning
    # Get individual letter dimensions for precise positioning
    r_bbox = draw.textbbox((0, 0), "R", font=font)
    r_width = r_bbox[2] - r_bbox[0]
    
    # Clear the previous text and redraw with precise positioning
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    
    # Draw R at the left edge
    r_x = 5  # Just barely touching left side
    r_y = y_position
    draw.text((r_x, r_y), "R", font=font, fill=(255, 255, 255, 255))
    
    # Draw L just merging with R (slight overlap)
    l_x = r_x + r_width - 20  # Overlap by 20 pixels
    l_y = y_position
    draw.text((l_x, l_y), "L", font=font, fill=(255, 255, 255, 255))
    
    # Apply macOS styling (rounded corners and padding)
    # Calculate the content size (80% of final size, leaving 20% for padding)
    content_size = int(final_size * 0.8)
    padding = (final_size - content_size) // 2
    
    # Resize the image to fit the content area
    img_resized = img.resize((content_size, content_size), Image.LANCZOS)
    
    # Create a new transparent image with the final size
    final_img = Image.new("RGBA", (final_size, final_size), (0, 0, 0, 0))
    
    # Paste the resized image in the center (creating the padding)
    final_img.paste(img_resized, (padding, padding), img_resized)
    
    # Create rounded corners
    corner_radius = int(final_size * 0.2)  # 20% of icon size
    mask = Image.new("L", (final_size, final_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, final_size, final_size], corner_radius, fill=255)
    
    # Apply the mask to create rounded corners
    final_img.putalpha(mask)
    
    # Save the result
    final_img.save('RamanLab_custom_rl.png', format="PNG")
    print("✅ Created RamanLab_custom_rl.png with custom RL positioning")
    
    # Also create a 512x512 version
    final_img_512 = final_img.resize((512, 512), Image.LANCZOS)
    final_img_512.save('RamanLab_custom_rl_512.png', format="PNG")
    print("✅ Created RamanLab_custom_rl_512.png")

if __name__ == "__main__":
    create_custom_rl_icon() 