from dataclasses import field

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def to_grey(image: np.ndarray) -> np.ndarray:
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    brightness = (0.299 * R + 0.5876 * G + 0.114 * B)

    return brightness


def add_noise_salt_pepper(image: np.ndarray, quality: float) -> Image.Image:
    noisy_array = np.copy(image)

    is_color = len(noisy_array.shape) == 3

    total_pixels = noisy_array.shape[0] * noisy_array.shape[1]

    num_salt = np.ceil(quality * total_pixels)
    num_pepper = np.ceil(quality * total_pixels)

    coords_salt = [np.random.randint(0, i, int(num_salt)) for i in noisy_array.shape[:2]]
    coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in noisy_array.shape[:2]]

    if is_color:
        noisy_array[coords_salt[0], coords_salt[1], :] = 255
        noisy_array[coords_pepper[0], coords_pepper[1], :] = 0
    else:
        noisy_array[coords_salt[0], coords_salt[1]] = 255
        noisy_array[coords_pepper[0], coords_pepper[1]] = 0

    return Image.fromarray(np.uint8(noisy_array))

def filter_median(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(image, ksize)

def filter_median_selfmade(image: np.ndarray, ksize: int) -> np.ndarray:
    H, W = image.shape[:2]

    output_array = np.zeros_like(image, dtype=np.float32)

    padding = ksize // 2

    for y in range(padding, H - padding):
        for x in range(padding, W - padding):
            window = image[y - padding: y + padding + 1,
                     x - padding: x + padding + 1]

            median_value = np.median(window, axis=(0, 1))

            output_array[y, x] = median_value

    return np.clip(output_array, 0, 255).astype(np.uint8)

def add_noise_gaussian(image: np.ndarray, sigma: float = 20) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def add_noise_line(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        temp_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        temp_img = image.copy()

    H, W = temp_img.shape[:2]

    is_horizontal = np.random.rand() > 0.5
    thickness = 1

    if is_horizontal:
        y = np.random.randint(0, H)
        pt1 = (0, y)
        pt2 = (W, y)
    else:
        x = np.random.randint(0, W)
        pt1 = (x, 0)
        pt2 = (x, H)

    cv2.line(temp_img, pt1, pt2, (255, 255, 255) , thickness)

    if len(image.shape) == 2:
        return cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    else:
        return temp_img

def average_filter(image_array: np.ndarray, ksize: int) -> np.ndarray:

    H, W = image_array.shape[:2]

    output_array = np.zeros_like(image_array, dtype=np.float32)

    padding = ksize // 2

    for y in range(padding, H - padding):
        for x in range(padding, W - padding):
            window = image_array[y - padding: y + padding + 1,
                     x - padding: x + padding + 1]

            average_value = np.mean(window, axis=(0, 1))

            output_array[y, x] = average_value

    return np.clip(output_array, 0, 255).astype(np.uint8)


def sharpen_laplacian_compare(image: np.ndarray, strength: float) -> np.ndarray:
    kernel = np.array([
        [0, -1, 0],
        [-1, strength, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # для корректной свертки в OpenCV нам нужно преобразовать rgb в bgr
    image_bgr = cv2.cvtColor(image.copy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    sharpened_image_bgr = cv2.filter2D(image_bgr, -1, kernel)

    # и обратно в rgb
    return cv2.cvtColor(sharpened_image_bgr, cv2.COLOR_BGR2RGB)

def apply_emboss(image: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image.copy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    kernel_emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)

    embossed_image_bgr = cv2.filter2D(image_bgr, -1, kernel_emboss)

    embossed_image_bgr = cv2.add(embossed_image_bgr, 70)

    return cv2.cvtColor(embossed_image_bgr, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    image = Image.open('example.jpg').convert('RGB')

    array = np.array(image, dtype=np.float32)

    grey_image = to_grey(array)

    #Image.fromarray(np.uint8(grey_image)).show()
    # quality от 0 до 1 (чем выше, тем хуже качество изображения)
    #noisy_grey_image = add_noise_salt_pepper(grey_image, 0.07)

    #noisy_grey_image_line = add_noise_line(np.array(noisy_grey_image, dtype=np.float32))
    #noisy_grey_image_line = add_noise_line(np.array(noisy_grey_image_line, dtype=np.float32))
    #noisy_grey_image_line = add_noise_line(np.array(noisy_grey_image_line, dtype=np.float32))

    #Image.fromarray(noisy_grey_image_line).show()

    #noisy_array_to_filter = np.array(noisy_grey_image_line, dtype=np.float32)

    # ksize - размер апертуры (то есть квадрата, который сканирует фотографию и убирает шум)
    #fixed_grey_image_median = filter_median_selfmade(noisy_array_to_filter, 5)
    #fixed_grey_image_median_and_average = Image.fromarray(average_filter(fixed_grey_image_median, 5))
    #fixed_grey_image_median_and_average.show()

    #to_sharp = np.array(fixed_grey_image_median_and_average, dtype=np.float32)
    #Image.fromarray(sharpen_laplacian_compare(to_sharp, 5)).show()

    Image.fromarray(apply_emboss(grey_image)).show()
