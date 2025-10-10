import pyglet
from pyglet.gl import *
from pyglet.window import key
import random
# Importamos nuestro back-end para el estado del cubo
from back_end.cube_model import RubikCube, solve_a_star, heuristic, solve_ida_star

# --- Variables Globales (Estética Final) ---
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
ROTATION_X = 35
ROTATION_Y = -25
CUBE_SIZE = 3.0     # Tamaño del cubie
GAP = 0.06          # Espacio entre cubies
ZOOM = -20.0        # Ajuste de zoom

# Lista de todos los movimientos posibles (para el revuelto)
ALL_MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]

# Mapeo de color del modelo a color RGB (RGBA: R, G, B, Alpha)
COLOR_MAP = {
    'W': (1.0, 1.0, 1.0, 1.0),                  # U: Blanco (White)
    'O': (1.0, 165/255.0, 0.0, 1.0),            # R: Naranja (Orange)
    'G': (0.0, 170/255.0, 0.0, 1.0),            # F: Verde (Green)
    'Y': (1.0, 1.0, 0.0, 1.0),                  # D: Amarillo (Yellow)
    'B': (0.0, 0.0, 1.0, 1.0),                  # L: Azul (Blue)
    'R': (1.0, 0.0, 0.0, 1.0),                  # B: Rojo (Red)
    'K': (0.1, 0.1, 0.1, 1.0),                  # Negro/Gris Oscuro (Cuerpo del cubie)
}

# La instancia de nuestro cubo de Rubik
rubik = RubikCube()
last_move_info = "Cubo resuelto"
solution_sequence = []
solution_step = 0


class RubikWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        # Configuración para antialiasing
        config = pyglet.gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
        try:
            super().__init__(*args, **kwargs, config=config)
        except pyglet.window.NoSuchConfigException:
            super().__init__(*args, **kwargs)
        
        self.set_caption("Rubik Solver IA (Pyglet)")
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        # Parámetros de la cámara
        self.rot_x = ROTATION_X
        self.rot_y = ROTATION_Y
        self.zoom = ZOOM
        
        pyglet.clock.schedule_interval(self.execute_solution_step, 0.5)

    def scramble_cube(self, num_moves=10):
        """Aplica una secuencia aleatoria de movimientos al cubo."""
        global rubik, last_move_info, solution_sequence, solution_step
        
        solution_sequence = []
        solution_step = 0
        
        """
        scramble_moves = [random.choice(ALL_MOVES) for _ in range(num_moves)]
        
        for move in scramble_moves:
            rubik = rubik.apply_move(move)
        """

        scramble_moves = []
        last_move_face = ''
        while len(scramble_moves) < num_moves:
            move = random.choice(ALL_MOVES)
            face = move[0]
        
            if face == last_move_face:
                continue
        
            rubik = rubik.apply_move(move) 
            scramble_moves.append(move)
            last_move_face = face

        scramble_str = " ".join(scramble_moves)
        last_move_info = f"Revuelto: {scramble_str}"
        h_score = heuristic(rubik)
        self.set_caption(f"Rubik Solver - Heurística: {h_score}") 
        print(f"Cubo revuelto con: {scramble_str}")
        print(f"Heurística inicial: {h_score}")

    def reset_cube_to_solved_state(self):
        """Reinicia el cubo a su estado solucionado y borra el historial de movimientos/solución."""
        global rubik, last_move_info, solution_sequence, solution_step
        
        # Re-inicializa el cubo a su estado resuelto (estado por defecto)
        rubik = RubikCube()
        last_move_info = "Cubo resuelto"
        solution_sequence = []
        solution_step = 0
        h_score = heuristic(rubik)
        self.set_caption(f"Rubik Solver - Heurística: {h_score}") 
        print("Cubo reiniciado al estado solucionado.")


    def on_resize(self, width, height):
        """Configura la proyección 3D al redimensionar."""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, float(width)/height, 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED
    
    def on_draw(self):
        """Dibuja el cubo y la interfaz 2D."""
        self.clear()
        glLoadIdentity()
        
        # 1. Configurar la cámara y la vista 3D
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0) # Rotación en X
        glRotatef(self.rot_y, 0.0, 1.0, 0.0) # Rotación en Y
        
        # Traslación para centrar el cubo (ajustada para CUBE_SIZE=3.0)
        center_offset = 1.5 * CUBE_SIZE + GAP * 3
        glTranslatef(-center_offset, -center_offset, -center_offset) 

        # 2. Dibujar los 27 Cubies (Iteración 3x3x3)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # Coordenadas del centro de cada cubie en el espacio 3D
                    center_x = x * CUBE_SIZE + x * GAP
                    center_y = y * CUBE_SIZE + y * GAP
                    center_z = z * CUBE_SIZE + z * GAP
                    
                    # Colores por defecto (caras no visibles)
                    color_u, color_d, color_f, color_b, color_l, color_r = ('K', 'K', 'K', 'K', 'K', 'K')

                    # Saltamos el cubie central (hueco)
                    if x == 1 and y == 1 and z == 1:
                        continue
                    
                    # --- Mapeo de coordenadas (x, y, z) a los índices del estado del cubo (0-8) ---
                    # El índice 0 es la esquina superior izquierda de la cara visible
                    
                    # Cara Superior (U): y = 2
                    if y == 2:
                        index_u = (z * 3) + x 
                        color_u = rubik.get_sticker_color('U', index_u)
                        
                    # Cara Inferior (D): y = 0
                    if y == 0:
                        index_d = ((2 - z) * 3) + x
                        color_d = rubik.get_sticker_color('D', index_d)
                        
                    # Cara Frontal (F): z = 2
                    if z == 2:
                        index_f = (2 - y) * 3 + x
                        color_f = rubik.get_sticker_color('F', index_f)

                    # Cara Trasera (B): z = 0
                    if z == 0:
                        index_b = (2 - y) * 3 + (2 - x) 
                        color_b = rubik.get_sticker_color('B', index_b)

                    # Cara Derecha (R): x = 2
                    if x == 2:
                        index_r = (2 - y) * 3 + (2 - z)
                        color_r = rubik.get_sticker_color('R', index_r)

                    # Cara Izquierda (L): x = 0
                    if x == 0:
                        index_l = (2 - y) * 3 + z
                        color_l = rubik.get_sticker_color('L', index_l)
                    
                    # Dibujar el cubie con sus colores
                    self.draw_cubie(
                        center_x, center_y, center_z, 
                        color_f, color_b, color_u, color_d, color_l, color_r
                    )

        # 3. Dibujar Botones y Texto 2D (Overlay)
        self.draw_2d_overlay()


    def draw_cubie(self, x, y, z, color_f, color_b, color_u, color_d, color_l, color_r):
        """
        Dibuja un solo cubie con sus 6 caras coloreadas, usando el factor de elevación
        para eliminar el Z-fighting.
        """
        glPushMatrix()
        
        # 1. Configuración de parámetros
        glTranslatef(x + CUBE_SIZE/2.0, y + CUBE_SIZE/2.0, z + CUBE_SIZE/2.0)
        
        s = CUBE_SIZE / 2.0  # Mitad del tamaño del cubie (para el cuerpo interior 'K')
        s_sticker = s * 0.98 # Tamaño ligeramente reducido para el sticker (mejor grosor de línea negra)
        
        # FACTOR DE ELEVACIÓN (Para evitar Z-fighting)
        ELEV = 0.01 
        
        # 2. Dibujar el cuerpo interno (Negro/Gris)
        glColor4f(*COLOR_MAP['K'])
        
        # Vertices para el cuerpo interno (tamaño completo 's')
        vertices_k = (
            # Front (Z+)
            s, s, s, -s, s, s, -s, -s, s, s, -s, s, 
            # Back (Z-)
            s, s, -s, s, -s, -s, -s, -s, -s, -s, s, -s, 
            # Up (Y+)
            s, s, s, s, s, -s, -s, s, -s, -s, s, s,  
            # Down (Y-)
            s, -s, s, -s, -s, s, -s, -s, -s, s, -s, -s, 
            # Left (X-)
            -s, s, s, -s, s, -s, -s, -s, -s, -s, -s, s, 
            # Right (X+)
            s, s, s, s, -s, s, s, -s, -s, s, s, -s
        )
        # Dibujar las 6 caras del cuerpo interno
        pyglet.graphics.draw(24, GL_QUADS, ('v3f', vertices_k))
            
        # 3. Dibujar los Stickers Coloreados
        
        # Vertices de las 6 caras (usando s_sticker y ELEV)
        
        # Front (Z+): Elevamos en Z
        vertices_f = ('v3f', (s_sticker, s_sticker, s + ELEV, -s_sticker, s_sticker, s + ELEV, -s_sticker, -s_sticker, s + ELEV, s_sticker, -s_sticker, s + ELEV)) 
        # Back (Z-): Reducimos en Z
        vertices_b = ('v3f', (-s_sticker, s_sticker, -s - ELEV, -s_sticker, -s_sticker, -s - ELEV, s_sticker, -s_sticker, -s - ELEV, s_sticker, s_sticker, -s - ELEV))
        # Up (Y+): Elevamos en Y
        vertices_u = ('v3f', (s_sticker, s + ELEV, s_sticker, s_sticker, s + ELEV, -s_sticker, -s_sticker, s + ELEV, -s_sticker, -s_sticker, s + ELEV, s_sticker))
        # Down (Y-): Reducimos en Y
        vertices_d = ('v3f', (s_sticker, -s - ELEV, s_sticker, s_sticker, -s - ELEV, -s_sticker, -s_sticker, -s - ELEV, -s_sticker, -s_sticker, -s - ELEV, s_sticker))
        # Left (X-): Reducimos en X
        vertices_l = ('v3f', (-s - ELEV, s_sticker, s_sticker, -s - ELEV, s_sticker, -s_sticker, -s - ELEV, -s_sticker, -s - ELEV, -s - ELEV, -s_sticker, s_sticker))
        # Right (X+): Elevamos en X
        vertices_r = ('v3f', (s + ELEV, s_sticker, -s_sticker, s + ELEV, s_sticker, s_sticker, s + ELEV, -s_sticker, s_sticker, s + ELEV, -s_sticker, -s_sticker))
        
        
        # Colores de las 6 caras
        colors = [color_f, color_b, color_u, color_d, color_l, color_r]
        vertices = [vertices_f, vertices_b, vertices_u, vertices_d, vertices_l, vertices_r]
        
        # Dibujar cada cara coloreada
        for i in range(6):
            if colors[i] != 'K':
                glColor4f(*COLOR_MAP[colors[i]])
                pyglet.graphics.draw(4, GL_QUADS, vertices[i])

        glPopMatrix()

    def draw_2d_overlay(self):
        """Dibuja el texto del overlay (Controles, estado y métricas de la IA)."""
        self.set_2d() 
        
        # Título y controles
        controls = [
            'Controles:',
            'Revuelto: [S] (Scramble - 8 movimientos)',
            'Resolver: [A] (A* Search)',
            'Resolver: [J] (Iterative Deepening A*)',
            'Resetear: [C] (Volver al estado solución)',
            'Rotaciones: U/Y, D/Z, F/G, B/P, L/I, R/T (Move / Move Prime)',
            f'Último Estado: {last_move_info}',
        ]

        for i, text in enumerate(controls):
            pyglet.text.Label(
                text,
                x=10, y=self.height - 20 - (i * 20),
                color=(255, 255, 255, 255),
                font_size=10
            ).draw()
            
        # Secuencia de Solución
        if solution_sequence:
            sol_text = f"Solución ({solution_step}/{len(solution_sequence)}): {' '.join(solution_sequence)}"
            pyglet.text.Label(
                sol_text,
                x=10, y=10,
                color=(0, 255, 0, 255) if solution_step >= len(solution_sequence) else (255, 255, 0, 255),
                font_size=12
            ).draw()

        self.set_3d() 

    def set_2d(self):
        """Configura la vista para dibujar elementos 2D."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def set_3d(self):
        """Restaura la vista 3D."""
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def execute_solution_step(self, dt):
        """Ejecuta un paso de la secuencia de solución cada 0.5 segundos."""
        global rubik, solution_sequence, solution_step, last_move_info

        if solution_step < len(solution_sequence):
            move = solution_sequence[solution_step]
            rubik = rubik.apply_move(move)
            last_move_info = f"Aplicando solución: {move}"
            solution_step += 1
            if solution_step == len(solution_sequence):
                last_move_info = "¡Cubo Resuelto por A*!"
                self.set_caption("Rubik Solver IA - ¡RESUELTO!")
                
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Permite rotar el cubo con el ratón."""
        if buttons & pyglet.window.mouse.LEFT:
            self.rot_y += dx * 0.5
            self.rot_x -= dy * 0.5


    def on_key_press(self, symbol, modifiers):
        """Maneja la aplicación de movimientos, el revuelto, el reset y la activación de A*."""
        global rubik, last_move_info, solution_sequence, solution_step

        move = None
        # Reiniciar la ejecución de la solución al hacer un movimiento manual o revuelto
        solution_sequence = []
        solution_step = 0 

        # --- 1. LÓGICA DE CONTROL (A*, Scramble y Reset) ---
        if symbol == key.A:
            if not rubik.is_solved():
                self.set_caption("Rubik Solver IA - BUSCANDO SOLUCIÓN...")
                solution, expanded_nodes = solve_a_star(rubik) 
                
                if solution:
                    solution_sequence = solution
                    last_move_info = f"IA: {len(solution)} movs, {expanded_nodes} nodos."
                    print(last_move_info)
                else:
                    last_move_info = f"IA: No se encontró solución ({expanded_nodes} nodos exp.)"
                    print(last_move_info)
            else:
                last_move_info = "El cubo ya está resuelto."
            return
        
        # Ejecuta IDA* y encola la solución para animar igual que A*
        elif symbol == key.J:
<<<<<<< Updated upstream
=======


        # Ejecuta IDA* y encola la solución para animar igual que A*
>>>>>>> Stashed changes
            if not rubik.is_solved():
                self.set_caption("Rubik Solver IA - IDA* BUSCANDO SOLUCIÓN...")
                solution, expanded_nodes = solve_ida_star(rubik)

                if solution:
                    self.set_caption(
                        f"Rubik Solver IA - IDA* listo: {len(solution)} movimientos | Nodos expandidos: {expanded_nodes}"
                    )
                    solution_sequence = [] # reinicia la cola por si venía de A*
                    solution_sequence.extend(solution)
                    solution_step = 0
                else:
                    self.set_caption(
                        f"Rubik Solver IA - IDA*: sin solución <= profundidad establecida | Nodos expandidos {expanded_nodes}"
                    )
            else:
                self.set_caption("Rubik Solver IA - Ya está resuelto")

        



        
        elif symbol == key.S:
<<<<<<< Updated upstream
            self.scramble_cube(num_moves = 8) # 8 movimientos para un buen revuelto inicial
=======
            self.scramble_cube(num_moves=8) # 10 movimientos para un buen revuelto inicial
>>>>>>> Stashed changes
            return

        elif symbol == key.C: # NUEVO: Tecla 'C' para Resetear
            self.reset_cube_to_solved_state()
            return
        


        # --- 2. CAPTURA DE TODOS LOS MOVIMIENTOS ---
        
        # R (Derecha) / R' (T)
        elif symbol == key.R: move = 'R'
        elif symbol == key.T: move = "R'"
            
        # L (Izquierda) / L' (I)
        elif symbol == key.L: move = 'L'
        elif symbol == key.I: move = "L'"
            
        # U (Arriba) / U' (Y)
        elif symbol == key.U: move = 'U'
        elif symbol == key.Y: move = "U'"
        
        # D (Abajo) / D' (Z)
        elif symbol == key.D: move = 'D'
        elif symbol == key.Z: move = "D'"
            
        # F (Frontal) / F' (G)
        elif symbol == key.F: move = 'F'
        elif symbol == key.G: move = "F'"
            
        # B (Trasera) / B' (P) 
        elif symbol == key.B: move = 'B'
        elif symbol == key.P: move = "B'"


        # --- 3. VERIFICACIÓN Y APLICACIÓN FINAL ---
        
        if move is None:
            return 
            
        # Aplicar el movimiento y actualizar el estado
        rubik = rubik.apply_move(move)
        h_score = heuristic(rubik)
        last_move_info = f"Movimiento: {move}. Heurística: {h_score}"
        self.set_caption(f"Rubik Solver - Heurística: {h_score}") 

# --- Ejecución ---
if __name__ == '__main__':
    try:
        window = RubikWindow(WINDOW_WIDTH, WINDOW_HEIGHT, resizable=True)
        glClearColor(0.1, 0.1, 0.1, 1.0) # Fondo Gris Oscuro
        glEnable(GL_DEPTH_TEST) # Habilita la profundidad para 3D
        glEnable(GL_CULL_FACE)  # Habilita el descarte de caras traseras
        pyglet.app.run()
    except Exception as e:
        print(f"Error al iniciar Pyglet: {e}")
        print("Asegúrate de tener Pyglet y sus dependencias (como OpenGL) instaladas.")
