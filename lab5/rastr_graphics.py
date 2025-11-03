import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pygame.transform import threshold


def calculate_brightness(array: np.ndarray) -> np.ndarray:
    # Y = 0,299R + 0,5876G + 0,114B
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]

    brightness = (0.299 * R + 0.5876 * G + 0.114 * B)

    return brightness.astype(np.uint8)

def generate_brightness_histogram(array: np.ndarray):
    brightness_array = calculate_brightness(array)

    # строим гистограмму с 256 ячейками (частота яркостей)
    hist, bins = np.histogram(brightness_array.flatten(), bins=256, range=[0, 256])

    plt.figure(figsize=(8, 4))
    plt.title("Гистограмма яркости")
    plt.xlabel("Уровень яркости (0-255)")
    plt.ylabel("Количество пикселей")

    # построение графика
    plt.bar(bins[:-1], hist, width=1, color='gray')
    plt.show()

def brightness_change(array: np.ndarray, value: int) -> Image.Image:
    # увеличение яркости
    brigter_array = array + value
    brigter_array = np.clip(brigter_array, 0, 255)
    brigter_pic = Image.fromarray(np.uint8(brigter_array))
    return brigter_pic

def contrast_change(array: np.ndarray, k: float) -> Image.Image:
    # изменение контрастности
    new_array = np.copy(array)

    # вычисляем среднюю яркость для каждого канала, np.mean(axis=(0, 1)) дает 3 значения (по одному для R, G, B)
    average_brightness = np.mean(new_array, axis=(0, 1))

    # применяем формулу ко всем каналам одновременно:
    new_array = k * (new_array - average_brightness) + average_brightness

    # контролируем диапазон цветов
    new_array = np.clip(new_array, 0, 255)
    return Image.fromarray(np.uint8(new_array))

def to_negative(array: np.ndarray) -> Image.Image:
    negative_array = 255 - array
    return Image.fromarray(np.uint8(negative_array))

def to_binary(array: np.ndarray) -> Image.Image:
    T = 127
    # Y = 0,299R + 0,5876G + 0,114B
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]

    brightness = (0.299 * R + 0.5876 * G + 0.114 * B)

    mask = brightness < T

    binary_array = np.zeros_like(array)  # изначально чёрный массив

    # инвертируем маску, устанавливаем белый цвет там где яркость больше порога
    binary_array[~mask] = 255

    return Image.fromarray(np.uint8(binary_array))

def to_grey(array: np.ndarray) -> Image.Image:
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]

    brightness = (0.299 * R + 0.5876 * G + 0.114 * B)

    return Image.fromarray(np.uint8(brightness))

if __name__ == '__main__':
    img = Image.open('example.jpg').convert('RGB')

    # но в конце нужно будет вернуться к np.uint8, но сейчас используем float32 чтобы избежать переполнения
    array = np.array(img, dtype=np.float32)


    print(f"Размер массива: {array.shape}")

    # генерация гистограммы
    generate_brightness_histogram(array)

    # преобразуем массив пикселей обратно в объект PIL для отображения и приводим обратно к uint8
    processed_img = Image.fromarray(np.uint8(array))
    processed_img.show()

    brightness_change(np.array(processed_img, dtype=np.float32), 100).show()
    contrast_change(np.array(processed_img, dtype=np.float32), 2).show()
    to_negative(np.array(processed_img, dtype=np.float32)).show()
    to_binary(np.array(processed_img, dtype=np.float32)).show()
    to_grey(np.array(processed_img, dtype=np.float32)).show()


