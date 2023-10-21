import os
import math
from PIL import Image
from fpdf import FPDF
import concurrent.futures

# Define parameters
folder_path = "Gal/train/Rings"
output_path= "Ringofmosaic.pdf"
thumbnail_size = (200, 200)
images_per_row = 4

# Get a list of all image files in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

# Calculate number of rows and columns needed
num_images = len(image_files)
num_columns = min(images_per_row, num_images)
num_rows = int(math.ceil(num_images / num_columns))

# Create a new PDF document that will hold the mosaic
pdf = FPDF(unit="pt", format=(thumbnail_size[0] * num_columns, thumbnail_size[1] * num_rows))
pdf.set_auto_page_break(0)

# Loop through all images and add them to the mosaic
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i, file in enumerate(image_files):
        # Open image and resize it
        image = Image.open(file)
        image.thumbnail(thumbnail_size)

        # Calculate the position of the image in the mosaic
        row = int(i / num_columns)
        col = i % num_columns
        x = col * thumbnail_size[0]
        y = row * thumbnail_size[1]

        # Add the image to the PDF document
        temp_image_file = f"temp_image_{i}.png"
        image.save(temp_image_file)
        futures.append(executor.submit(pdf.add_page))
        futures.append(executor.submit(pdf.image, temp_image_file, x, y))

    # Wait for all futures to complete before continuing
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
        except Exception as exc:
            print(f"Error: {exc}")

# Save the PDF document
pdf.output(output_path, "F")

# Remove temporary image files
for i in range(num_images):
    temp_image_file = f"temp_image_{i}.png"
    os.remove(temp_image_file)
