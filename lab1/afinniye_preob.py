import pygame
import numpy as np
import math

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Аффинные преобразования 3D объекта")
clock = pygame.time.Clock()

import numpy as np

def create_letter(depth=12):
    out_front_points = []
    in_front_points = []
    out_back_points = []
    in_back_points = []

    for angle_deg in range(60, 301, 15):
        angle_rad = np.radians(angle_deg)
        x_out = 70 * np.cos(angle_rad)
        y_out = 70 * np.sin(angle_rad)
        x_in = 40 * np.cos(angle_rad)
        y_in = 40 * np.sin(angle_rad)

        out_front_points.append([x_out, y_out, depth, 1])
        in_front_points.append([x_in, y_in, depth, 1])
        out_back_points.append([x_out, y_out, -depth, 1])
        in_back_points.append([x_in, y_in, -depth, 1])

    vertices = np.array(
        out_front_points + in_front_points + out_back_points + in_back_points
    )
    return vertices

vertices = create_letter()
n = len(vertices) // 4  # количество точек на одной полуокружности

edges = []

for i in range(n - 1):
    edges.append((i, i + 1))

for i in range(n, 2 * n - 1):
    edges.append((i, i + 1))

for i in range(2 * n, 3 * n - 1):
    edges.append((i, i + 1))

for i in range(3 * n, 4 * n - 1):
    edges.append((i, i + 1))

for i in range(n):
    edges.append((i, i + 2 * n))

for i in range(n, 2 * n):
    edges.append((i, i + 2 * n))

# дополнительные боковые соединения между внешним и внутренним полукругом
# передняя грань
for i in range(n - 1):
    edges.append((i, i + n))

# задняя грань
for i in range(2 * n, 3 * n - 1):
    edges.append((i, i + n))

# замыкание контура
edges.append((n - 1, n - 1 + n))         # передняя грань
edges.append((3 * n - 1, 3 * n - 1 + n)) # задняя грань


# матрица переноса
def create_translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1]
    ])


# матрица вращения
def create_rotation_matrix_x(angle):
    rad = math.radians(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(rad), math.sin(rad), 0],
        [0, -math.sin(rad), math.cos(rad), 0],
        [0, 0, 0, 1]
    ])


def create_rotation_matrix_y(angle):
    rad = math.radians(angle)
    return np.array([
        [math.cos(rad), 0, -math.sin(rad), 0],
        [0, 1, 0, 0],
        [math.sin(rad), 0, math.cos(rad), 0],
        [0, 0, 0, 1]
    ])


def create_rotation_matrix_z(angle):
    rad = math.radians(angle)
    return np.array([
        [math.cos(rad), math.sin(rad), 0, 0],
        [-math.sin(rad), math.cos(rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


# матрица масштабирования
def create_scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


# матрица ортографической проекции на плоскость XY
projection_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])

# основной цикл и переменные
running = True
# параметры преобразований
rotation_angle_x, rotation_angle_y, rotation_angle_z = 0, 0, 0
translation_x, translation_y, translation_z = 0, 0, 0
scale_factor = 1.0

# вариант 5. Вращение по спирали вдоль осей с замедлением.
animation_active = False
animation_rotation_speed = 2.0
animation_translation_speed = 1.0
damping_factor = 0.98

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # управление клавиатурой
            if event.key == pygame.K_UP:
                translation_y -= 10
            if event.key == pygame.K_DOWN:
                translation_y += 10
            if event.key == pygame.K_LEFT:
                translation_x -= 10
            if event.key == pygame.K_RIGHT:
                translation_x += 10
            if event.key == pygame.K_PAGEUP:
                translation_z += 10
            if event.key == pygame.K_PAGEDOWN:
                translation_z -= 10
            if event.key == pygame.K_KP_PLUS or event.key == pygame.K_PLUS:
                scale_factor *= 1.1
            if event.key == pygame.K_KP_MINUS or event.key == pygame.K_MINUS:
                scale_factor /= 1.1
            if event.key == pygame.K_s:  # запуск/остановка анимации
                animation_active = not animation_active
                if not animation_active:
                    animation_rotation_speed = 2.0
                    animation_translation_speed = 1.0

    # управление мышью
    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        dx, dy = pygame.mouse.get_rel()
        rotation_angle_y += dx * 0.5
        rotation_angle_x += dy * 0.5

    # логика анимаци
    if animation_active:
        # вращение по спирали вдоль оси Y
        rotation_angle_y += animation_rotation_speed
        translation_y -= animation_translation_speed
        # замедление
        animation_rotation_speed *= damping_factor
        animation_translation_speed *= damping_factor

        # остановка анимации при достижении минимальной скорости
        if abs(animation_rotation_speed) < 0.01 and abs(animation_translation_speed) < 0.01:
            animation_active = False

    # обновление матриц
    translation_mat = create_translation_matrix(translation_x, translation_y, translation_z)
    rotation_mat_x = create_rotation_matrix_x(rotation_angle_x)
    rotation_mat_y = create_rotation_matrix_y(rotation_angle_y)
    rotation_mat_z = create_rotation_matrix_z(rotation_angle_z)
    scaling_mat = create_scaling_matrix(scale_factor, scale_factor, scale_factor)

    # комбинированная матрица
    model_matrix = scaling_mat @ rotation_mat_x @ rotation_mat_y @ rotation_mat_z @ translation_mat

    screen.fill((0, 0, 0))  # чёрный фон

    transformed_vertices = vertices @ model_matrix @ projection_matrix

    # смещение для центрирования объекта
    offset_x = width / 2
    offset_y = height / 2

    # отрисовка рёбер
    for edge in edges:
        p1_index, p2_index = edge
        p1 = transformed_vertices[p1_index]
        p2 = transformed_vertices[p2_index]

        # проекция и отрисовка
        pygame.draw.line(screen, (255, 255, 255),
                         (p1[0] + offset_x, p1[1] + offset_y),
                         (p2[0] + offset_x, p2[1] + offset_y), 2)

    pygame.display.flip()
    clock.tick(60)  # 60 fps

pygame.quit()