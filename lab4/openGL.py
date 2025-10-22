from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
import time
from PIL import Image


class TextureManager:
    def __init__(self):
        self.textures = {}

    def load_texture(self, filename, texture_id):
        try:
            image = Image.open(filename)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()

            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

            self.textures[filename] = texture_id
            print(f"Загружена текстура: {filename} -> ID: {texture_id}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки текстуры {filename}: {e}")
            return False

    def get_texture_id(self, filename):
        return self.textures.get(filename)


class Camera:
    def __init__(self):
        self.position = np.array([0.0, 3.0, 8.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 0.3
        self.mouse_sensitivity = 0.1
        self.last_mouse_x = 400
        self.last_mouse_y = 300
        self.first_mouse = True

    def process_keyboard(self, key):
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        front = front / np.linalg.norm(front)

        right = np.cross(front, np.array([0.0, 1.0, 0.0]))
        right = right / np.linalg.norm(right)

        if key == b'w':
            self.position += front * self.speed
        elif key == b's':
            self.position -= front * self.speed
        elif key == b'a':
            self.position -= right * self.speed
        elif key == b'd':
            self.position += right * self.speed
        elif key == b'q':
            self.position[1] -= self.speed
        elif key == b'e':
            self.position[1] += self.speed

        # Ограничиваем высоту полёта камеры
        if self.position[1] < 0.5:
            self.position[1] = 0.5

    def process_mouse(self, x, y):
        if self.first_mouse:
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.first_mouse = False

        x_offset = x - self.last_mouse_x
        y_offset = self.last_mouse_y - y

        self.last_mouse_x = x
        self.last_mouse_y = y

        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

    def apply(self):
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        front = front / np.linalg.norm(front)
        target = self.position + front

        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            target[0], target[1], target[2],
            0, 1, 0
        )


class SkyDome:
    def __init__(self, radius=50.0):
        self.radius = radius

    def draw(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("sky.png"))

        # Отключаем освещение для неба
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)

        slices = 32
        stacks = 16

        for i in range(stacks):
            lat0 = math.pi * (-0.5 + (i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = math.pi * (-0.5 + (i + 1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)

            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * (j) / slices
                x = math.cos(lng)
                y = math.sin(lng)

                # Текстурные координаты
                s0 = j / slices
                t0 = i / stacks
                s1 = j / slices
                t1 = (i + 1) / stacks

                glTexCoord2f(s0, t0)
                glVertex3f(x * zr0 * self.radius, z0 * self.radius, y * zr0 * self.radius)

                glTexCoord2f(s1, t1)
                glVertex3f(x * zr1 * self.radius, z1 * self.radius, y * zr1 * self.radius)
            glEnd()

        glEnable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)


class House:
    def __init__(self, x, z, width=1.2, height=2.0, depth=1.2):
        self.position = np.array([x, 0.0, z])
        self.width = width
        self.height = height
        self.depth = depth

    def draw(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("brick.png"))

        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 10.0)

        # Основание дома
        self.draw_textured_cube(self.width, self.height, self.depth)

        # Крыша
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("roof.png"))
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])

        glPushMatrix()
        glTranslatef(0, self.height / 2, 0)
        glScalef(self.width * 1.2, self.height * 0.6, self.depth * 1.2)
        self.draw_textured_pyramid()
        glPopMatrix()

        # Окна
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("window.png"))
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 0.5, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.5, 0.7, 1.0])

        # Окно 1
        glPushMatrix()
        glTranslatef(0, self.height * 0.3, self.depth / 2 + 0.01)
        glScalef(0.4, 0.5, 0.1)
        self.draw_textured_quad()
        glPopMatrix()

        # Окно 2
        glPushMatrix()
        glTranslatef(self.width / 2 + 0.01, self.height * 0.3, 0)
        glScalef(0.1, 0.5, 0.4)
        self.draw_textured_quad()
        glPopMatrix()

        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

    def draw_textured_cube(self, w, h, d):
        # Передняя грань
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w / 2, -h / 2, d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(w / 2, -h / 2, d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(w / 2, h / 2, d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(-w / 2, h / 2, d / 2)
        glEnd()

        # Задняя грань
        glBegin(GL_QUADS)
        glNormal3f(0, 0, -1)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w / 2, -h / 2, -d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(-w / 2, h / 2, -d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(w / 2, h / 2, -d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(w / 2, -h / 2, -d / 2)
        glEnd()

        # Верхняя грань
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w / 2, h / 2, -d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(-w / 2, h / 2, d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(w / 2, h / 2, d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(w / 2, h / 2, -d / 2)
        glEnd()

        # Нижняя грань
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w / 2, -h / 2, -d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(w / 2, -h / 2, -d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(w / 2, -h / 2, d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(-w / 2, -h / 2, d / 2)
        glEnd()

        # Правая грань
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(w / 2, -h / 2, -d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(w / 2, h / 2, -d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(w / 2, h / 2, d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(w / 2, -h / 2, d / 2)
        glEnd()

        # Левая грань
        glBegin(GL_QUADS)
        glNormal3f(-1, 0, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w / 2, -h / 2, -d / 2)
        glTexCoord2f(2.0, 0.0);
        glVertex3f(-w / 2, -h / 2, d / 2)
        glTexCoord2f(2.0, 2.0);
        glVertex3f(-w / 2, h / 2, d / 2)
        glTexCoord2f(0.0, 2.0);
        glVertex3f(-w / 2, h / 2, -d / 2)
        glEnd()

    def draw_textured_pyramid(self):
        # Боковые грани
        glBegin(GL_TRIANGLES)
        # Передняя грань
        glNormal3f(0, 0.5, 0.5)
        glTexCoord2f(0.5, 1.0);
        glVertex3f(0, 0.5, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-0.5, -0.5, 0.5)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(0.5, -0.5, 0.5)

        # Правая грань
        glNormal3f(0.5, 0.5, 0)
        glTexCoord2f(0.5, 1.0);
        glVertex3f(0, 0.5, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(0.5, -0.5, 0.5)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(0.5, -0.5, -0.5)

        # Задняя грань
        glNormal3f(0, 0.5, -0.5)
        glTexCoord2f(0.5, 1.0);
        glVertex3f(0, 0.5, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(-0.5, -0.5, -0.5)

        # Левая грань
        glNormal3f(-0.5, 0.5, 0)
        glTexCoord2f(0.5, 1.0);
        glVertex3f(0, 0.5, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(-0.5, -0.5, 0.5)
        glEnd()

        # Основание пирамиды
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-0.5, -0.5, 0.5)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(0.5, -0.5, 0.5)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(0.5, -0.5, -0.5)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-0.5, -0.5, -0.5)
        glEnd()

    def draw_textured_quad(self):
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-0.5, -0.5, 0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(0.5, -0.5, 0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(0.5, 0.5, 0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-0.5, 0.5, 0)
        glEnd()


class Road:
    def __init__(self, length=25.0):
        self.length = length

    def draw(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("asphalt.png"))

        # Асфальт
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])

        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-2.5, 0.01, -self.length / 2)
        glTexCoord2f(5.0, 0.0);
        glVertex3f(2.5, 0.01, -self.length / 2)
        glTexCoord2f(5.0, 5.0);
        glVertex3f(2.5, 0.01, self.length / 2)
        glTexCoord2f(0.0, 5.0);
        glVertex3f(-2.5, 0.01, self.length / 2)
        glEnd()

        glDisable(GL_TEXTURE_2D)

        # Разметка на дороге (белая)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.7, 0.7, 0.6, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.9, 0.9, 0.8, 1.0])

        for z in np.arange(-self.length / 2 + 1, self.length / 2, 3):
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(-0.15, 0.02, z - 1.0)
            glVertex3f(0.15, 0.02, z - 1.0)
            glVertex3f(0.15, 0.02, z + 1.0)
            glVertex3f(-0.15, 0.02, z + 1.0)
            glEnd()


class Scene:
    def __init__(self):
        self.camera = Camera()
        self.houses = []
        self.road = Road()
        self.sky_dome = SkyDome()
        self.setup_houses()

    def setup_houses(self):
        # Дома слева от дороги
        for z in np.arange(-10, 11, 4):
            self.houses.append(House(-5, z, 1.5, 2.2, 1.8))
            self.houses.append(House(-8, z + 2, 1.3, 1.8, 1.6))

        # Дома справа от дороги
        for z in np.arange(-10, 11, 4):
            self.houses.append(House(5, z, 1.4, 2.0, 1.7))
            self.houses.append(House(8, z + 2, 1.6, 2.4, 1.9))

    def draw_ground(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_manager.get_texture_id("earth.png"))

        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])

        size = 30
        segments = 10

        glBegin(GL_QUADS)
        for i in range(segments):
            for j in range(segments):
                x0 = -size + (i * 2 * size / segments)
                z0 = -size + (j * 2 * size / segments)
                x1 = x0 + 2 * size / segments
                z1 = z0 + 2 * size / segments

                glNormal3f(0, 1, 0)
                glTexCoord2f(i / 2, j / 2);
                glVertex3f(x0, 0, z0)
                glTexCoord2f((i + 1) / 2, j / 2);
                glVertex3f(x1, 0, z0)
                glTexCoord2f((i + 1) / 2, (j + 1) / 2);
                glVertex3f(x1, 0, z1)
                glTexCoord2f(i / 2, (j + 1) / 2);
                glVertex3f(x0, 0, z1)
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def draw(self):
        # Небо
        self.sky_dome.draw()

        # Земля
        self.draw_ground()

        # Дорога
        self.road.draw()

        # Дома
        for house in self.houses:
            house.draw()


def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Источник света
    glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 10.0, 0.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)


def setup_textures():
    global texture_manager
    texture_manager = TextureManager()

    texture_ids = glGenTextures(6)

    # Загружаем текстуры
    texture_manager.load_texture("brick.png", texture_ids[0])
    texture_manager.load_texture("window.png", texture_ids[1])
    texture_manager.load_texture("asphalt.png", texture_ids[2])
    texture_manager.load_texture("roof.png", texture_ids[3])
    texture_manager.load_texture("earth.png", texture_ids[4])
    texture_manager.load_texture("sky.png", texture_ids[5])


# Глобальные переменные
scene = None
texture_manager = None
window_width, window_height = 1200, 800


def init():
    global scene
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glClearColor(0.7, 0.8, 1.0, 1.0)

    glEnable(GL_TEXTURE_2D)

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    setup_lighting()
    setup_textures()
    scene = Scene()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    scene.camera.apply()

    scene.draw()

    glutSwapBuffers()


def reshape(width, height):
    global window_width, window_height
    window_width, window_height = width, height
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def keyboard(key, x, y):
    if key == b'\x1b':
        exit()
    else:
        scene.camera.process_keyboard(key)
        glutPostRedisplay()


def mouse_motion(x, y):
    scene.camera.process_mouse(x, y)
    glutPostRedisplay()


def special_keys(key, x, y):
    if key == GLUT_KEY_UP:
        scene.camera.position[1] += scene.camera.speed
    elif key == GLUT_KEY_DOWN:
        scene.camera.position[1] -= scene.camera.speed
    glutPostRedisplay()


def idle():
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1200, 800)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Gorodok")

    init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutPassiveMotionFunc(mouse_motion)
    glutSpecialFunc(special_keys)
    glutIdleFunc(idle)

    glutSetCursor(GLUT_CURSOR_NONE)

    glutWarpPointer(window_width // 2, window_height // 2)

    glutMainLoop()


if __name__ == "__main__":
    main()