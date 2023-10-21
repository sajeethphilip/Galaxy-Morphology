import os
from PIL import Image

# set path to image folder and output filename
image_folder = "Gal/train/Rings"
output_filename = "WithRings.png"

# get list of image filenames in folder
image_filenames = os.listdir(image_folder)
num_images = len(image_filenames)

# calculate number of rows and columns based on number of images
if num_images <= 4:
    rows, cols = 2, 2
elif num_images <= 6:
    rows, cols = 2, 3
elif num_images <= 9:
    rows, cols = 3, 3
else:
    rows, cols = 4, 4

# set width and height of each image tile in mosaic
monitor_width, monitor_height = 1920, 1080 # set monitor resolution
tile_width = monitor_width // cols
tile_height = monitor_height // rows

# create new image for mosaic
mosaic_width = tile_width * cols
mosaic_height = tile_height * rows
mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

# loop through images and paste into mosaic
for i, filename in enumerate(image_filenames):
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).resize((tile_width, tile_height))
    row = i // cols
    col = i % cols
    x = col * tile_width
    y = row * tile_height
    mosaic.paste(image, (x, y))

# save mosaic
mosaic.save(output_filename)
