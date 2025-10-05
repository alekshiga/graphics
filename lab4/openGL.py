import contextlib, sys
import ctypes
from asyncio import log

import numpy as np
from OpenGL import GL as gl
import glfw

def perspective(near, far, fov, aspect):
    n = near
    f = far

    t = np.tan((fov * np.pi / 180) / 2) * near
    b = -t
    r = t * aspect
    l = b * aspect

    if abs(n - f) <= 0:
        raise ValueError("near и far должны быть существенно разными.")


    return np.array((
        ((2 * n) / (r - l), 0, 0, 0),
        (0, (2 * n) / (t - b), 0, 0),
        ((r + l) / (r - l), (t + b) / (t - b), (f + n) / (n - f), -1),
        (0, 0, 2 * f * n / (n - f), 0)))


def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def look_at(eye, target, up):
    zax = normalized(eye - target)
    xax = normalized(np.cross(up, zax))
    yax = np.cross(zax, xax)

    x = - xax.dot(eye)
    y = - yax.dot(eye)
    z = - zax.dot(eye)

    return np.array(((xax[0], yax[0], zax[0], 0),
                     (xax[1], yax[1], zax[1], 0),
                     (xax[2], yax[2], zax[2], 0),
                     (x, y, z, 1)))


def create_mvp(program_id, width, height):
    fov = 60
    near = 0.01
    far = 100.0
    eye = np.array([2, 3, 3])
    target, up = np.array((0, 0, 0)), np.array((0, 1, 0))

    projection = perspective(near, far, fov, width / height)
    view = look_at(eye, target, up)
    model = np.identity(4)

    mvp = model @ view @ projection  # Умножение M * V * P
    matrix_id = gl.glGetUniformLocation(program_id, 'MVP')
    return matrix_id, mvp.astype(np.float32)


@contextlib.contextmanager
def load_shaders():
    shaders = {
        gl.GL_VERTEX_SHADER: '''\
                #version 330 core
                layout(location = 0) in vec3 vertexPosition_modelspace;
                layout(location = 1) in vec3 vertexColor_modelspace;

                uniform mat4 MVP; 

                out vec3 fragmentColor;

                void main(){
                  gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0); 
                  fragmentColor = vertexColor_modelspace;
                }
                ''',
        gl.GL_FRAGMENT_SHADER: '''\
                #version 330 core

                in vec3 fragmentColor;
                out vec3 color;

                void main(){
                  color = fragmentColor;
                } 
                '''
    }

    program_id = gl.glCreateProgram()
    try:
        shader_ids = []
        for shader_type, shader_src in shaders.items():
            shader_id = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader_id, shader_src)

            gl.glCompileShader(shader_id)

            result = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
            info_log_len = gl.glGetShaderiv(shader_id, gl.GL_INFO_LOG_LENGTH)

            if info_log_len and not result:
                logmsg = gl.glGetShaderInfoLog(shader_id)
                log.error("Ошибка компиляции шейдера:\n" + str(logmsg))
                sys.exit(10)

            gl.glAttachShader(program_id, shader_id)
            shader_ids.append(shader_id)

        gl.glLinkProgram(program_id)

        result = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        info_log_len = gl.glGetProgramiv(program_id, gl.GL_INFO_LOG_LENGTH)
        if info_log_len and not result:
            logmsg = gl.glGetProgramInfoLog(program_id)
            log.error("Ошибка линковки программы:\n" + str(logmsg))
            sys.exit(11)

        gl.glUseProgram(program_id)
        yield program_id
    finally:
        for shader_id in shader_ids:
            gl.glDetachShader(program_id, shader_id)
            gl.glDeleteShader(shader_id)
        gl.glUseProgram(0)
        gl.glDeleteProgram(program_id)


@contextlib.contextmanager
def create_vertex_array_object():
    vertex_array_id = gl.glGenVertexArrays(1)
    try:
        gl.glBindVertexArray(vertex_array_id)
        yield
    finally:
        gl.glDeleteVertexArrays(1, [vertex_array_id])

@contextlib.contextmanager
def create_vertex_buffer():
    attr_id_position = 0
    attr_id_color = 1
    vertex_buffer = 0  # ID буфера позиций
    color_buffer = 0  # ID буфера цветов

    with create_vertex_array_object():
        try:
            vertex_buffer = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
            array_type = (ctypes.c_float * len(g_vertex_buffer_data))
            gl.glBufferData(gl.GL_ARRAY_BUFFER,
                            len(g_vertex_buffer_data) * ctypes.sizeof(ctypes.c_float),
                            array_type(*g_vertex_buffer_data),
                            gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(
                attr_id_position,
                3,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                0,
                None
            )
            gl.glEnableVertexAttribArray(attr_id_position)

            color_buffer = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_buffer)

            array_type_color = (ctypes.c_float * len(g_color_buffer_data))
            gl.glBufferData(gl.GL_ARRAY_BUFFER,
                            len(g_color_buffer_data) * ctypes.sizeof(ctypes.c_float),
                            array_type_color(*g_color_buffer_data),
                            gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(
                attr_id_color,
                3,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                0,
                None
            )
            gl.glEnableVertexAttribArray(attr_id_color)

            yield

        finally:
            gl.glDisableVertexAttribArray(attr_id_position)
            gl.glDisableVertexAttribArray(attr_id_color)
            gl.glDeleteBuffers(1, [vertex_buffer]) # очистка буфера вершин
            gl.glDeleteBuffers(1, [color_buffer])  # очистка цветового буфера
             # x, y, z
g_vertex_buffer_data = [0, 0, 0,
               0, -1, 0,
               -1, 0, 0, # 1

               -1, -1, 0,
               0, -1, 0,
               -1, 0, 0, # 2 # треугольники нижней грани

               0, 0, -1,
               -1, 0, -1,
               0, -1, -1, # 3

               -1, -1, -1,
               0, -1, -1,
               -1, 0, -1, # 4 # треугольники верхной грани

               0, 0, 0,
               0, -1, 0,
               0, -1, -1, # 5

               0, -1, -1,
               0, -1, 0,
               0, 0, -1, # 6 # треугольники левой грани

               -1, -1, -1,
               -1, -1, 0,
               -1, 0, 0, # 7

               -1, 0, -1,
               -1, -1, -1,
               -1, 0, 0, # 8 # треугольники правой грани

                0, 0, 0,
               -1, 0, 0,
               -1, 0, -1, # 9

               -1, 0, -1,
               0, 0, 0,
               0, 0, -1, # 10 # треугольники передней грани

               -1, -1, -1,
               -1, -1, 0,
               0, -1, 0, # 11

               -1, -1, -1,
               0, -1, 0,
               0, -1, 0] # 12 # треугольники задней грани

g_color_buffer_data = np.array([
    0.583,  0.771,  0.014,
    0.609,  0.115,  0.436,
    0.327,  0.483,  0.844,
    0.822,  0.569,  0.201,
    0.435,  0.602,  0.223,
    0.310,  0.747,  0.402,
    0.716,  0.738,  0.370,
    0.302,  0.459,  0.589,
    0.583,  0.771,  0.014,
    0.609,  0.115,  0.436,
    0.327,  0.483,  0.844,
    0.822,  0.569,  0.201,
    0.435,  0.602,  0.223,
    0.310,  0.747,  0.402,
    0.716,  0.738,  0.370,
    0.302,  0.459,  0.589,
    0.583,  0.771,  0.014,
    0.609,  0.115,  0.436,
    0.327,  0.483,  0.844,
    0.822,  0.569,  0.201,
    0.435,  0.602,  0.223,
    0.310,  0.747,  0.402,
    0.716,  0.738,  0.370,
    0.302,  0.459,  0.589,
    0.583,  0.771,  0.014,
    0.609,  0.115,  0.436,
    0.327,  0.483,  0.844,
    0.822,  0.569,  0.201,
    0.435,  0.602,  0.223,
    0.310,  0.747,  0.402,
    0.716,  0.738,  0.370,
    0.302,  0.459,  0.589,
    0.583,  0.771,  0.014,
    0.609,  0.115,  0.436,
    0.327,  0.483,  0.844,
    0.822,  0.569,  0.201
], dtype=np.float32)


@contextlib.contextmanager
def create_main_window():
    if not glfw.init():
        sys.exit(1)
    try:
        title = 'Colored Cube'
        window = glfw.create_window(640, 480, title, None, None)
        if not window:
            sys.exit(2)
        glfw.make_context_current(window)

        glfw.set_input_mode(window, glfw.STICKY_KEYS, True)
        gl.glClearColor(0, 0, 0.4, 0)

        # удаление невидимых граней
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_CULL_FACE)

        yield window

    finally:
        glfw.terminate()


def main_loop(window, mvp_id, mvp):
    while (
            glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and
            not glfw.window_should_close(window)
    ):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUniformMatrix4fv(mvp_id, 1, False, mvp)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 12 * 3)

        glfw.swap_buffers(window)
        glfw.poll_events()


if __name__ == '__main__':
    width, height = 640, 480
    with create_main_window() as window:
        with create_vertex_buffer():
            with load_shaders() as prog_id:
                mvp_id, mvp = create_mvp(prog_id, width, height)
                main_loop(window, mvp_id, mvp)