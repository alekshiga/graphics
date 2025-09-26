import pygame
import numpy as np
import math
import time

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Аффинные преобразования 3D объекта")
clock = pygame.time.Clock()

def create_c_vertices(depth=12):
    outer_points_front = []
    inner_points_front = []
    outer_points_back = []
    inner_points_back = []

    for angle_deg in range(60, 301, 15):
        angle_rad = np.radians(angle_deg)
        x_outer = 70 * np.cos(angle_rad)
        y_outer = 70 * np.sin(angle_rad)
        x_inner = 40 * np.cos(angle_rad)
        y_inner = 40 * np.sin(angle_rad)

        outer_points_front.append([x_outer, y_outer, depth, 1])
        inner_points_front.append([x_inner, y_inner, depth, 1])
        outer_points_back.append([x_outer, y_outer, -depth, 1])
        inner_points_back.append([x_inner, y_inner, -depth, 1])

    vertices = np.array(
        outer_points_front + inner_points_front + outer_points_back + inner_points_back
    )
    return vertices

vertices = create_c_vertices()
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

# матрица вращения вокруг осей
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

axis_vertices = np.array([
    [0, 0, 0, 1], # начало координат
    [100, 0, 0, 1], # ось X
    [0, 100, 0, 1], # ось Y
    [0, 0, 100, 1], # ось Z
])

running = True

# параметры преобразований
rotation_angle_x, rotation_angle_y, rotation_angle_z = 0, 0, 0
translation_x, translation_y, translation_z = 0, 0, 0
scale_factor = 1.0

animation_active = False
initial_rotation_speed = 3
initial_translation_speed = 5
decay_factor = 0.955 # уменьшает скорость в каждом кадре

current_rotation_speed_x = 2
current_rotation_speed_y = 2
current_rotation_speed_z = 2

current_translation_speed_x = 10
current_translation_speed_y = 0
current_translation_speed_z = 0

rotation_speed = 180
radius_x, radius_y, radius_z = 50, 30, 20

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
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
            if event.key == pygame.K_s:
                animation_active = not animation_active
                if animation_active:
                    animation_start_time = time.time()

    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        dx, dy = pygame.mouse.get_rel()
        rotation_angle_y += dx * 0.5
        rotation_angle_x += dy * 0.5

    if animation_active:
        rotation_angle_y += current_rotation_speed_y
        rotation_angle_x += current_rotation_speed_x
        rotation_angle_z += current_rotation_speed_z

        translation_x += current_translation_speed_x
        translation_y += current_translation_speed_y
        translation_z += current_translation_speed_z

        current_rotation_speed_x *= decay_factor
        current_rotation_speed_y *= decay_factor
        current_rotation_speed_z *= decay_factor

        current_translation_speed_x *= decay_factor
        current_translation_speed_y *= decay_factor
        current_translation_speed_z *= decay_factor

        if (abs(current_rotation_speed_x) < 0.05 and
                abs(current_rotation_speed_y) < 0.05 and
                abs(current_translation_speed_x) < 0.05 and
                abs(current_translation_speed_y) < 0.05):
            animation_active = False

    translation_mat = create_translation_matrix(translation_x, translation_y, translation_z)
    rotation_mat_x = create_rotation_matrix_x(rotation_angle_x)
    rotation_mat_y = create_rotation_matrix_y(rotation_angle_y)
    rotation_mat_z = create_rotation_matrix_z(rotation_angle_z)
    scaling_mat = create_scaling_matrix(scale_factor, scale_factor, scale_factor)

    model_matrix = scaling_mat @ rotation_mat_x @ rotation_mat_y @ rotation_mat_z @ translation_mat

    screen.fill((0, 0, 0))

    transformed_vertices = vertices @ model_matrix @ projection_matrix

    offset_x = width / 2
    offset_y = height / 2

    for edge in edges:
        p1_index, p2_index = edge
        p1 = transformed_vertices[p1_index]
        p2 = transformed_vertices[p2_index]

        pygame.draw.line(screen, (255, 255, 255),
                         (p1[0] + offset_x, p1[1] + offset_y),
                         (p2[0] + offset_x, p2[1] + offset_y), 2)

        transformed_axes = axis_vertices @ projection_matrix

        # отображение осей разными цветами
        origin = (transformed_axes[0][0] + offset_x, transformed_axes[0][1] + offset_y)
        x_axis = (transformed_axes[1][0] + offset_x, transformed_axes[1][1] + offset_y)
        y_axis = (transformed_axes[2][0] + offset_x, transformed_axes[2][1] + offset_y)
        z_axis = (transformed_axes[3][0] + offset_x, transformed_axes[3][1] + offset_y)

        pygame.draw.line(screen, (255, 0, 0), origin, x_axis, 2)  # ось X - красная
        pygame.draw.line(screen, (0, 255, 0), origin, y_axis, 2)  # ось Y - зеленая
        pygame.draw.line(screen, (0, 0, 255), origin, z_axis, 2)  # ось Z - синяя

        font = pygame.font.SysFont('Arial', 18, bold=True)
        text_x = font.render('X', True, (255, 0, 0))
        text_y = font.render('Y', True, (0, 255, 0))
        text_z = font.render('Z', True, (0, 0, 255))

        screen.blit(text_x, x_axis)
        screen.blit(text_y, y_axis)
        screen.blit(text_z, z_axis)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
