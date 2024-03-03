import glob
import os
from PIL import ImageFont, ImageDraw, Image

# Constants
DIR_PATH = "C:/Reza/FunSpace/Digital_Generated_Images/single_language_English/"
FONTS_PATH = "training_data_fonts/1English/*"
IMAGE_SIZES = [32,58,64,128,256,512]
IMG_SETUP = {
    32: (28, 30, 2),
    58: (49, 50, 5),
    64: (53, 55, 5),
    128: (135, 140, 5),
    256: (198, 200, 5),
    512: (430, 431, 5)
}

def get_fonts():
    # Read the font-adjusted.txt file and return a list of characters
    with open('font-adjusted.txt', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    output = [list(line.split("_")[0]) for line in lines]
    return output

def create_image(char, font_path, size, range_start, range_end, constant):
    # Create an image of a character with a specific font and size
    try:
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size)
        text_size = draw.textsize(char, font=font)
        x = (size - text_size[0]) // 2
        y = (size - text_size[1]) // 2 - constant
        draw.text((x, y), char, 255, font=font)
        return img
    except Exception as e:
        print(f"Failed to create image for character {char} with font {font_path} and size {size}: {e}")
        return None

def main():
    # Main function to generate images of characters in different fonts and sizes
    output = get_fonts()
    fonts = glob.glob(FONTS_PATH)
    for size in IMAGE_SIZES:
        os.makedirs(f'{DIR_PATH}{size}x{size}', exist_ok=True)
        range_start, range_end, constant = IMG_SETUP[size]
        for i, char_row in enumerate(output):
            os.makedirs(f'{DIR_PATH}{size}x{size}/{i}', exist_ok=True)
            for font_path in fonts:
                for char in char_row:
                    for font_size in range(range_start, range_end):
                        img = create_image(char, font_path, font_size, range_start, range_end, constant)
                        if img is not None:
                            img.save(f'{DIR_PATH}{size}x{size}/{i}/{char}_{font_size}.jpg')

if __name__ == "__main__":
    main()
