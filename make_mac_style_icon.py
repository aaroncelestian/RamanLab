from PIL import Image, ImageDraw
import os

def make_mac_style_icon(input_path, output_path, final_size=512):
    """
    Convert an existing icon to macOS style with:
    - 20% transparent padding around the edges
    - Rounded corners (20% of the icon size)
    - Proper square aspect ratio
    """
    
    # Open the original image
    img = Image.open(input_path).convert("RGBA")
    
    # Calculate the content size (80% of final size, leaving 20% for padding)
    content_size = int(final_size * 0.8)
    padding = (final_size - content_size) // 2
    
    # Resize the original image to fit the content area
    img_resized = img.resize((content_size, content_size), Image.LANCZOS)
    
    # Create a new transparent image with the final size
    final_img = Image.new("RGBA", (final_size, final_size), (0, 0, 0, 0))
    
    # Paste the resized image in the center (creating the padding)
    final_img.paste(img_resized, (padding, padding), img_resized)
    
    # Create rounded corners
    corner_radius = int(final_size * 0.2)  # 20% of icon size
    mask = Image.new("L", (final_size, final_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, final_size, final_size], corner_radius, fill=255)
    
    # Apply the mask to create rounded corners
    final_img.putalpha(mask)
    
    # Save the result
    final_img.save(output_path, format="PNG")
    print(f"Created macOS-style icon: {output_path}")

# Convert the existing RamanLab icon
if os.path.exists("RamanLab_1024.png"):
    # Use the largest PNG version for best quality
    make_mac_style_icon("RamanLab_1024.png", "RamanLab_mac_style.png", 512)
elif os.path.exists("RamanLab_512.png"):
    make_mac_style_icon("RamanLab_512.png", "RamanLab_mac_style.png", 512)
else:
    print("No suitable RamanLab PNG found. Please ensure RamanLab_1024.png or RamanLab_512.png exists.")

# Also create multiple sizes for a complete icon set
sizes = [16, 32, 64, 128, 256, 512, 1024]
for size in sizes:
    if os.path.exists("RamanLab_1024.png"):
        make_mac_style_icon("RamanLab_1024.png", f"RamanLab_mac_{size}.png", size)
        print(f"Created {size}x{size} macOS-style icon") 