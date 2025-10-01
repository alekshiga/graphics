import contextlib, sys
import ctypes
from asyncio import log

from OpenGL import GL as gl
import glfw

@contextlib.contextmanager
def load_shaders():
    shaders = {
        gl.GL_VERTEX_SHADER: '''\
                #version 330 core
                layout(location = 0) in vec3 vertexPosition_modelspace;
                void main(){
                  gl_Position.xyz = vertexPosition_modelspace;
                  gl_Position.w = 1.0;
                }
                ''',
        gl.GL_FRAGMENT_SHADER: '''\
                #version 330 core
                out vec3 color;
                void main(){
                  color = vec3(1,0,1);
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

            if info_log_len:
                logmsg = gl.glGetShaderInfoLog(shader_id)
                log.error(logmsg)
                sys.exit(10)

            gl.glAttachShader(program_id, shader_id)
            shader_ids.append(shader_id)

        gl.glLinkProgram(program_id)

        # check if linking was successful
        result = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        info_log_len = gl.glGetProgramiv(program_id, gl.GL_INFO_LOG_LENGTH)
        if info_log_len:
            logmsg = gl.glGetProgramInfoLog(program_id)
            log.error(logmsg)
            sys.exit(11)

        gl.glUseProgram(program_id)
        yield
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
    with create_vertex_array_object():
        # первый треугольник
        vertex_data = [-1, -1, 0,
                       1, -1, 0,
                       0, 1, 0]

        attr_id = 0

        vertex_buffer = gl.glGenBuffers(1)
        try:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
            array_type = (ctypes.c_float * len(vertex_data))
            gl.glBufferData(gl.GL_ARRAY_BUFFER,
                            len(vertex_data) * ctypes.sizeof(ctypes.c_float),
                            array_type(*vertex_data),
                            gl.GL_STATIC_DRAW)

            # читай по 3 числа типа float для каждой вершины
            gl.glVertexAttribPointer(
                attr_id, # id атрибута
                3,  # размер (x,y,z)
                gl.GL_FLOAT, # тип данных float
                gl.GL_FALSE,
                0,   # шаг (0 - плотная упаковка)
                None  # смещение откуда начинать читать (0)
            )
            gl.glEnableVertexAttribArray(attr_id)
            yield
        finally:
            gl.glDisableVertexAttribArray(attr_id)
            gl.glDeleteBuffers(1, [vertex_buffer])


@contextlib.contextmanager
def create_main_window():
    if not glfw.init():
        sys.exit(1)
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        title = 'first experience with OpenGL'
        window = glfw.create_window(640, 480, title, None, None)
        if not window:
            sys.exit(2)
        glfw.make_context_current(window)

        glfw.set_input_mode(window, glfw.STICKY_KEYS, True)
        gl.glClearColor(0, 0, 0.4, 0)

        yield window

    finally:
        glfw.terminate()

def main_loop(window):
    while (
        glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and
        not glfw.window_should_close(window)
    ):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # рисование треугольника
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        # отображение кадра и обработка событий
        glfw.swap_buffers(window)
        glfw.poll_events()

if __name__ == '__main__':
    # создаём окно
    with create_main_window() as window:
        # создаём VAO/VBO и настраиваем атрибуты
        with create_vertex_buffer():
            # комплириуем, линкуем, активируем шейдеры
            with load_shaders():
                # запуск цикла рендеринга с готовыми ресурсами
                main_loop(window)
                # удаляются шейдеры, удаляются VAO/VBO