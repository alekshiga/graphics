import math
import numpy as np
import pygame

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Освещение")
clock = pygame.time.Clock()

# кривая, из которой потом получится тело похожее на вазу
curved_line = [(y, 50) for y in np.linspace(-100, 100, 20)]

# создадим фигуру путём вращения кривой вокруг оси Y
def create_rotation_figure(curve):
    segment_count = 30
    vertices = []
    triangles = []
    number_points_in_curve = len(curve)

    for i in range(0, segment_count + 1):
        angle = 2 * math.pi * i / segment_count
        for y, r in curve:
            x = r * math.cos(angle)
            z = r * math.sin(angle)
            vertices.append(np.array([x, y, z, 1], dtype=np.float64))

    # триангуляция
    for i in range(segment_count):
        for j in range(number_points_in_curve - 1):
            # вершины для текущего кольца
            idx1 = i * number_points_in_curve + j
            idx2 = idx1 + number_points_in_curve

            # вершины для следующего кольца
            idx3 = idx1 + 1
            idx4 = idx2 + 1

            # первый треугольник
            triangles.append([idx1, idx2, idx3])
            # второй треугольник
            triangles.append([idx2, idx4, idx3])

    return np.array(vertices), triangles

vertices, triangles = create_rotation_figure(curved_line)

# матрицы преобразований (из лр1)
def create_translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1]
    ], dtype=np.float64)


def create_rotation_matrix_x(angle):
    rad = math.radians(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(rad), math.sin(rad), 0],
        [0, -math.sin(rad), math.cos(rad), 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def create_rotation_matrix_y(angle):
    rad = math.radians(angle)
    return np.array([
        [math.cos(rad), 0, -math.sin(rad), 0],
        [0, 1, 0, 0],
        [math.sin(rad), 0, math.cos(rad), 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def create_rotation_matrix_z(angle):
    rad = math.radians(angle)
    return np.array([
        [math.cos(rad), math.sin(rad), 0, 0],
        [-math.sin(rad), math.cos(rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def create_scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


projection_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


running = True
rotation_angle_x, rotation_angle_y, rotation_angle_z = 0, 0, 0
translation_x, translation_y, translation_z = 0, 0, 0
scale_factor = 1.0

rotation_speed = 180
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

    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        dx, dy = pygame.mouse.get_rel()
        rotation_angle_y += dx * 0.5
        rotation_angle_x += dy * 0.5

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

    # сортируем треугольники по Z (для правильного закрашивания невидимой части тела)
    sorted_triangles = sorted(triangles, key=lambda tri: -np.mean(transformed_vertices[tri, 2]))
    screen.fill((0, 0, 0))

    transformed_vertices = vertices @ model_matrix
    projected_vertices = transformed_vertices @ projection_matrix

    for tri in triangles:
        v1_idx, v2_idx, v3_idx = tri

        p1_proj = projected_vertices[v1_idx]
        p2_proj = projected_vertices[v2_idx]
        p3_proj = projected_vertices[v3_idx]

        p1_screen = (p1_proj[0] + offset_x, p1_proj[1] + offset_y)
        p2_screen = (p2_proj[0] + offset_x, p2_proj[1] + offset_y)
        p3_screen = (p3_proj[0] + offset_x, p3_proj[1] + offset_y)

        pygame.draw.polygon(screen, (255, 255, 255), [p1_screen, p2_screen, p3_screen], 1)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()