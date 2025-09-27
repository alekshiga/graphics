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
def create_rotation_figure_with_caps(curve):
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

    # центры крышек
    bottom_center_idx = len(vertices)
    top_center_idx = bottom_center_idx + 1

    bottom_y = curve[0][0]
    bottom_r = curve[0][1]
    vertices = np.vstack([vertices, np.array([0, bottom_y, 0, 1])])  # центр низа

    top_y = curve[-1][0]
    top_r = curve[-1][1]
    vertices = np.vstack([vertices, np.array([0, top_y, 0, 1])])  # центр верха

    for i in range(segment_count):
        idx1 = i * number_points_in_curve  # нижнее кольцо
        idx2 = ((i + 1) % segment_count) * number_points_in_curve
        triangles.append([bottom_center_idx, idx1, idx2])

    for i in range(segment_count):
        idx1 = i * number_points_in_curve + (number_points_in_curve - 1)  # верхнее кольцо
        idx2 = ((i + 1) % segment_count) * number_points_in_curve + (number_points_in_curve - 1)
        triangles.append([top_center_idx, idx2, idx1])


    return np.array(vertices), triangles

vertices, triangles = create_rotation_figure_with_caps(curved_line)

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

def calculate_normals(vertices, triangles):
    normals = np.zeros(vertices.shape, dtype=np.float64)
    normals = normals[:, :3]  # учитываем только XYZ

    for tri in triangles:
        v0 = vertices[tri[0]][:3]
        v1 = vertices[tri[1]][:3]
        v2 = vertices[tri[2]][:3]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal) + 1e-10  # нормализация, +1e-10 чтобы избежать деления на 0

        for idx in tri:
            normals[idx] += normal

    # нормализуем все нормали вершин
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
    normals /= norms
    return normals


def phong_lighting(vertex_pos, normal, light_pos, camera_pos, ka=0.1, kd=0.7, ks=0.2, shininess=32):
    L = light_pos - vertex_pos
    L /= np.linalg.norm(L)

    V = camera_pos - vertex_pos
    V /= np.linalg.norm(V)

    N = normal
    R = 2 * np.dot(N, L) * N - L
    R /= np.linalg.norm(R)

    ambient = ka

    diff = max(np.dot(N, L), 0)
    diffuse = kd * diff

    spec = max(np.dot(R, V), 0) ** shininess
    specular = ks * spec

    intensity = ambient + diffuse + specular
    intensity = np.clip(intensity, 0, 1)
    color = int(255 * intensity)
    return (color, color, color)

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

    transformed_vertices = vertices @ model_matrix

    normals = calculate_normals(transformed_vertices, triangles)

    projected_vertices = transformed_vertices @ projection_matrix

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

        #pygame.draw.polygon(screen, (255, 255, 255), [p1_screen, p2_screen, p3_screen], 1)

        light_pos = np.array([100, 200, -100], dtype=np.float64)
        camera_pos = np.array([0, 0, -500], dtype=np.float64)

        # вершины в 3D координатах
        p1_world = transformed_vertices[v1_idx][:3]
        p2_world = transformed_vertices[v2_idx][:3]
        p3_world = transformed_vertices[v3_idx][:3]

        edge1 = p2_world - p1_world
        edge2 = p3_world - p1_world
        face_normal = np.cross(edge1, edge2)
        face_normal /= np.linalg.norm(face_normal) + 1e-10  # нормализация

        view_vec = camera_pos - p1_world

        if np.dot(face_normal, view_vec) <= 0:
            continue

            # находим нормали к каждой вершине треугольника
        n1 = normals[v1_idx]
        n2 = normals[v2_idx]
        n3 = normals[v3_idx]

        # закраска фонга
        c1 = phong_lighting(p1_world, n1, light_pos, camera_pos)
        c2 = phong_lighting(p2_world, n2, light_pos, camera_pos)
        c3 = phong_lighting(p3_world, n3, light_pos, camera_pos)

        color = (
            (c1[0] + c2[0] + c3[0]),
            (c1[1] + c2[1] + c3[1]),
            (c1[2] + c2[2] + c3[2]),
        )

        # проекция на XY
        p1_proj = projected_vertices[v1_idx]
        p2_proj = projected_vertices[v2_idx]
        p3_proj = projected_vertices[v3_idx]

        p1_screen = (int(p1_proj[0] + offset_x), int(p1_proj[1] + offset_y))
        p2_screen = (int(p2_proj[0] + offset_x), int(p2_proj[1] + offset_y))
        p3_screen = (int(p3_proj[0] + offset_x), int(p3_proj[1] + offset_y))

        # закрашиваем треугольники средними цветами
        pygame.draw.polygon(screen, color, [p1_screen, p2_screen, p3_screen])

    pygame.display.flip()
    clock.tick(60)

pygame.quit()