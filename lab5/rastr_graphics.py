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

def to_negative(array: np.ndarray, region: tuple = None) -> Image.Image:
    negative_array = np.copy(array)
    if region:
        x_min, y_min, x_max, y_max = region
        negative_array[y_min:y_max, x_min:x_max, :] = 255 - array[y_min:y_max, x_min:x_max, :]
    else:
        negative_array = 255 - array
    return Image.fromarray(np.uint8(negative_array))

def to_binary(array: np.ndarray) -> Image.Image:
    # Y = 0,299R + 0,5876G + 0,114B
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]

    brightness = (0.299 * R + 0.5876 * G + 0.114 * B)

    # вычислим среднюю яркость, это и будет порогом, а потом посмотрим различия
    T = np.mean(brightness)
    print(f"Порог = {T:.2f}")

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

# определит порог для бинаризации
def otsu_T(array: np.ndarray) -> int:
    brightness_array = calculate_brightness(array).astype(np.uint8).flatten()
    # строим гистограмму и нормализуем её до вероятностей
    # _ используем по соглашению Python для переменной, т.к. её значение нам не нужно
    hist, _ = np.histogram(brightness_array, bins=256, range=[0, 256])
    p = hist / brightness_array.size

    best_T = 0
    max_dispersion = 0.0

    for T in range(1, 255):
        # w0 - вероятность(доля) фона
        # w1 - вероятность(доля) объекта
        w0 = np.sum(p[:T])
        w1 = np.sum(p[T:])

        # пропускаем итерацию цикла, чтобы пропустить критические пороги (например T = 1),
        # т.к. если на фото не будет абсолютно чёрных пикселей, возникнет деление на 0
        if w0 == 0 or w1 == 0:
            continue

        # вычисляем уровни яркости от 0 до T-1 и от T до 255
        mu0 = np.sum(np.arange(T) * p[:T]) / w0
        mu1 = np.sum(np.arange(T, 256) * p[T:]) / w1

        # вычисление дисперсии
        dispersion = w0 * w1 * (mu0 - mu1) ** 2

        if dispersion > max_dispersion:
            max_dispersion = dispersion
            best_T = T

    return best_T


def to_binary_otsu(array: np.ndarray) -> Image.Image:

    T_otsu = otsu_T(array)
    print(f"Порог по Оцу = {T_otsu}")

    brightness_array = calculate_brightness(array)
    mask = brightness_array < T_otsu
    binary_array = np.zeros_like(array)
    binary_array[~mask] = 255

    return Image.fromarray(np.uint8(binary_array))

if __name__ == '__main__':
    img = Image.open('example.jpg').convert('RGB')

    # но в конце нужно будет вернуться к np.uint8, но сейчас используем float32 чтобы избежать переполнения
    array = np.array(img, dtype=np.float32)


    print(f"Размер массива: {array.shape}")

    # генерация гистограммы
    generate_brightness_histogram(array)


    brightness_change(np.array(img, dtype=np.float32), 100).show()
    contrast_change(np.array(img, dtype=np.float32), 2).show()
    # to_binary(np.array(img, dtype=np.float32)).show()
    to_binary_otsu(np.array(img, dtype=np.float32)).show()
    to_grey(np.array(img, dtype=np.float32)).show()

    region = (300, 300, 800, 600)
    to_negative(np.array(img, dtype=np.float32), region).show()