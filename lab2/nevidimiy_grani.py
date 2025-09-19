import pygame
import numpy as np
import math

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Удаление невидимых повeрхностей")
clock = pygame.time.Clock()

# (x, y, z, 1)
polygons = [
    # квадрат 1 синий
    np.array([
        [-50, -50, 50, 1],
        [50, -50, 50, 1],
        [50, 50, 50, 1],
        [-50, 50, 50, 1],
    ], dtype=float),

    # квадрат 2 зеленый
    np.array([
        [-70, -70, -50, 1],
        [30, -70, -50, 1],
        [30, 30, -50, 1],
        [-70, 30, -50, 1],
    ], dtype=float),

    # треугольник 1 красный
    np.array([
        [-60, 180, 0, 1],
        [60, 180, 0, 1],
        [0, 80, 0, 1],
    ], dtype=float),

    # квадрат 3 желтый
    np.array([
        [-40, -40, 0, 1],
        [40, -40, 0, 1],
        [40, 40, 0, 1],
        [-40, 40, 0, 1],
    ], dtype=float),

    # треугольник 2 фиолетовый
    np.array([
        [-20, -60, 20, 1],
        [20, -60, 20, 1],
        [0, 50, -30, 1],
    ], dtype=float),
]

# цвета многоугольников
colors = [
    (0, 0, 255, 255),  # синий
    (0, 255, 0, 255),  # зеленый
    (255, 0, 0, 255),  # красный
    (255, 255, 0, 255),  # желтый
    (128, 0, 128, 255),  # фиолетовый
]


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


"""функция, которая вычисляет Z-координату для точки на плоскости многоугольника,
 берёт 3 точки, строит 2 вектора на плоскости и находит их произведение, получая третий,
 перпендикулярный плоскости вектор, тем самым определяя координату z, чтобы потом определить,
 какая фигура ближе к экрану"""

def get_z_on_plane(p1, p2, p3, x, y):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    A, B, C = cross_product
    D = -np.dot(cross_product, p1)

    if C == 0:
        return np.inf

    return -(A * x + B * y + D) / C


# функция для проверки, находится ли точка внутри многоугольника
def is_point_in_polygon(x, y, poly_points):
    num_vertices = len(poly_points)
    is_inside = False

    p1x, p1y = poly_points[0]
    for i in range(num_vertices + 1):
        p2x, p2y = poly_points[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        is_inside = not is_inside
        p1x, p1y = p2x, p2y

    return is_inside


running = True
rotation_angle_x, rotation_angle_y, rotation_angle_z = 0, 0, 0
translation_x, translation_y, translation_z = 0, 0, 0
scale_factor = 1.0

rotation_speed = 180
radius_x, radius_y, radius_z = 50, 30, 20

# обработка событий как в лр1
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

    """модель-видовая матрица для преобразования 3D-координат в 2D изображение,
    получается путём перемножения модельной матрицы на матрицу вида, то есть 
    расположение объектов в пространстве, их размер и т.д. а также поставить камеру так, 
    чтобы объекты оказались прямо перед нами"""
    transformed_polygons_3d = []
    for polygon in polygons:
        transformed_polygons_3d.append(polygon @ model_matrix)

    projected_polygons = [poly @ projection_matrix for poly in transformed_polygons_3d]

    screen.fill((0, 0, 0))

    """реализация интервального алгоритма построчного сканирования,
    в данном случае строка сканирования - 800px"""

    offset_x = width / 2
    offset_y = height / 2

    for y in range(height):
        intersections = []
        for i, poly in enumerate(projected_polygons):
            for j in range(len(poly)):
                p1_proj = poly[j]
                p2_proj = poly[(j + 1) % len(poly)]

                p1_y_screen = p1_proj[1] + offset_y
                p2_y_screen = p2_proj[1] + offset_y

                if (p1_y_screen <= y < p2_y_screen) or (p2_y_screen <= y < p1_y_screen):
                    if p2_y_screen != p1_y_screen:
                        x_intersect = p1_proj[0] + (y - p1_y_screen) * (p2_proj[0] - p1_proj[0]) / (
                                    p2_y_screen - p1_y_screen)
                        intersections.append({'x': x_intersect, 'poly_idx': i})

        intersections.sort(key=lambda item: item['x'])

        i = 0
        while i < len(intersections):
            x_start = intersections[i]['x']

            j = i + 1
            while j < len(intersections) and intersections[j]['x'] == x_start:
                j += 1

            x_end = intersections[j]['x'] if j < len(intersections) else x_start

            if x_end > x_start:
                mid_x = (x_start + x_end) / 2

                closest_z = np.inf
                closest_poly_idx = -1

                for poly_idx in range(len(transformed_polygons_3d)):
                    poly_2d = projected_polygons[poly_idx][:, :2]

                    if is_point_in_polygon(mid_x, y - offset_y, poly_2d):
                        poly_3d = transformed_polygons_3d[poly_idx]
                        p1, p2, p3 = poly_3d[0][:3], poly_3d[1][:3], poly_3d[2][:3]

                        z_val = get_z_on_plane(p1, p2, p3, mid_x, y - offset_y)

                        if z_val < closest_z:
                            closest_z = z_val
                            closest_poly_idx = poly_idx

                if closest_poly_idx != -1:
                    color = colors[closest_poly_idx]
                    pygame.draw.line(screen, color,
                                     (x_start + offset_x, y),
                                     (x_end + offset_x, y), 1)

            i = j

    pygame.display.flip()
    clock.tick(60)

pygame.quit()