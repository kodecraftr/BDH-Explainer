import re
import base64
import os
from bs4 import BeautifulSoup

def extract_images_to_public(html_path, output_dir="public"):
    # Ensure the public directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find all image tags that use base64 data URIs
    images = soup.find_all('img', src=re.compile(r'^data:image/'))
    
    for i, img in enumerate(images):
        src = img['src']
        
        # Regex to separate the image extension and the base64 data
        match = re.match(r'data:image/(?P<ext>[a-zA-Z0-9\+]+);base64,(?P<data>.*)', src)
        if not match:
            continue
            
        ext = match.group('ext')
        if ext == 'svg+xml':  # Handle inline SVGs correctly
            ext = 'svg'
            
        img_data = match.group('data')
        
        # Generate a clean filename
        filename = f"bdh_asset_{i}.{ext}"
        filepath = os.path.join(output_dir, filename)

        # Decode and save the binary image data
        with open(filepath, 'wb') as img_file:
            img_file.write(base64.b64decode(img_data))

        # Update the HTML element to point to the new relative path
        img['src'] = f"{output_dir}/{filename}"

    # Write the cleaned-up HTML to a new file
    updated_html_path = html_path.replace('.html', '-updated.html')
    with open(updated_html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
        
    print(f"Extraction complete. {len(images)} images saved to '{output_dir}/'.")
    print(f"Refactored code saved as: {updated_html_path}")

# Run the extraction
if __name__ == "__main__":
    extract_images_to_public('bdh-implementation.html')
