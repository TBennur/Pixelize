# Generic
import os
import pathlib
import ctypes

# Image Generation
import torch
from diffusers import StableDiffusionPipeline

# Palette Generation
import torchvision.transforms.functional as TF
from selenium import webdriver
import base64

# Stylizing
import numpy as np
from PIL import Image

# Change these to alter generation
NUM_COLORS = 8
QUERY = "High Fantasy"
PROMPT = "a photo of an astronaut riding a horse on mars"
COUNT = 5

file_path = pathlib.Path("..")

def get_file_path(stem):
    for path in file_path.rglob(stem):
        return str(path)

stylize = ctypes.CDLL(get_file_path("website/stylizer/stylize.dll")).stylize
stylize.restype = None
stylize.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int16),
                    np.ctypeslib.ndpointer(ctypes.c_int16), 
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_bool]


model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# From Online
def driversetup():
    options = webdriver.ChromeOptions()
    #run Selenium in headless mode
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    #overcome limited resource problems
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("lang=en")
    #open Browser in maximized mode
    options.add_argument("start-maximized")
    #disable infobars
    options.add_argument("disable-infobars")
    #disable extension
    options.add_argument("--disable-extensions")
    options.add_argument("--incognito")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('log-level=3')
    driver = webdriver.Chrome(options=options)

    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

    return driver

# Creating a webdriver instance
driver = driversetup()
driver.maximize_window()
 
# Open Google Images in the browser
search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={QUERY}"
images_url = []

def get_images():
    # open browser and begin search
    driver.get(search_url)
    images = driver.find_element(webdriver.common.by.By.CLASS_NAME, 'islrc')
    elements = images.find_elements(webdriver.common.by.By.CLASS_NAME, 'rg_i')

    count = 1
    for e in elements:
        # get images source url
        img = e.get_attribute('src')
        with open(f"images/{QUERY}{count}.png", "wb") as fh:
            fh.write(base64.b64decode(img[22:]))
        
        count += 1
        if count > COUNT:
            break

# Gets K-Means starting value
def get_starting_values(image_tensor):

    values = []
    starting_values = torch.FloatTensor(NUM_COLORS, 3)
    i = 0

    while len(values) < NUM_COLORS:
        should_add_palette = True
        for value in values:
            if image_tensor[i, 0] == value[0] and image_tensor[i, 1] == value[1] and image_tensor[i, 2] == value[2]:
                should_add_palette = False
        if should_add_palette:
            values.append(image_tensor[i])
            starting_values[len(values) - 1] = values[len(values) - 1]    
        i += 1
        
    return starting_values

# Runs K-means
def KMeans(image_tensor, num_iterations = 25):

    N, D = image_tensor.shape

    dominant_colors = get_starting_values(image_tensor)

    image_tensor_flattened = image_tensor.view(N, 1, D)
    dominant_colors_flattened_j = dominant_colors.view(1, NUM_COLORS, D)

    for _ in range(num_iterations):

        distances = ((image_tensor_flattened - dominant_colors_flattened_j) ** 2).sum(-1)
        closest_colors = distances.argmin(dim=1).long().view(-1)

        dominant_colors.zero_()
        dominant_colors.scatter_add_(0, closest_colors[:, None].repeat(1, D), image_tensor)
        
        pixels_per_color = torch.bincount(closest_colors, minlength = NUM_COLORS).type_as(dominant_colors).view(NUM_COLORS, 1)
        dominant_colors /= pixels_per_color

    return dominant_colors.to(torch.int16).detach().numpy()

# Wrapper for k-means and color generation
def find_dominant_colors():
    
    full_tensor = torch.empty((1,3))
    for image_name in os.listdir('./images'):
        image = Image.open('./images/' + image_name)
        image_tensor = TF.to_tensor(image)
        image_tensor.unsqueeze_(0)
        image_tensor = torch.flatten(torch.flatten(image_tensor, 2, 3), 0, 1).transpose(0, 1)
        os.remove('./images/' + image_name)
        try:
            full_tensor = torch.cat((full_tensor, image_tensor), 0)
        except:
            pass

    return KMeans(256 * full_tensor)


# Main stylization function
def convert_image(image_name, palette):
    img = Image.open(image_name)
    img.load()    
    data = np.asarray(img, dtype = "int16")[:, :, 0:3]
    dimensions = (len(data), len(data[0]))
    palette = palette
    stylize(data, palette, dimensions[0], dimensions[1], NUM_COLORS, image_name.endswith(".jpg"))
    return Image.fromarray(np.uint8(data), 'RGB')


image = pipe(PROMPT).images[0]  
image.save(f"{PROMPT}.jpg")

get_images()
palette = find_dominant_colors()
final_image = convert_image(f"./{PROMPT}.jpg", palette)
final_image.save(f"{PROMPT}({QUERY}).jpg")

driver.close()