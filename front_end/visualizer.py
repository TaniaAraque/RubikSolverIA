# front_end/visualizer.py
import pyglet
from pyglet.gl import *
from pyglet.window import key
# Importamos nuestro back-end para el estado del cubo
from back_end.cube_model import RubikCube, solve_a_star

# --- Variables Globales ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ROTATION_X = -30
ROTATION_Y = -30
CUBE_SIZE = 1.0 # Tamaño de un cubie
GAP = 0.05      # Espacio entre cubies (opcional)

# Mapeo de color del modelo a color RGB para Pyglet
COLOR_MAP = {
    'W': (1.0, 1.0, 1.0, 1.0), # Blanco
    'Y': (1.0, 1.0, 0.0, 1.0), # Amarillo
    'G': (0.0, 1.0, 0.0, 1.0), # Verde
    'B': (0.0, 0.0, 1.0, 1.0), # Azul
    'R': (1.0, 0.0, 0.0, 1.0), # Rojo
    'O': (1.0, 0.6, 0.0, 1.0), # Naranja
    'K': (0.1, 0.1, 0.1, 1.0), # Negro/Gris Oscuro
}

# La instancia de nuestro cubo de Rubik
rubik = RubikCube()

class RubikWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_caption("Rubik Solver IA (Pyglet)")
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        # Parámetros de la cámara
        self.rot_x = ROTATION_X
        self.rot_y = ROTATION_Y
        self.zoom = -10.0

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, float(width)/height, 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED
    
    def on_draw(self):
        self.clear()
        glLoadIdentity()
        
        # 1. Configurar la cámara y la vista 3D
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0) # Rotación en X
        glRotatef(self.rot_y, 0.0, 1.0, 0.0) # Rotación en Y
        
        # Centrar el cubo 3x3. El cubo va de -1.5 a 1.5 en cada eje
        glTranslatef(-1.5 * CUBE_SIZE, -1.5 * CUBE_SIZE, -1.5 * CUBE_SIZE) 

        # 2. Dibujar los 27 Cubies
        # Iteramos sobre la cuadrícula 3x3x3. Las coordenadas (0, 1, 2)
        # se mapean al centro de cada cubie.
        
        # Mapeo de índices de estado a posiciones (Esto es una simplificación)
        # La lógica de qué color poner en qué cara del cubie es compleja. 
        
        # Para simplificar la lógica de color, iteraremos 3x3x3 cubies.
        # Solo dibujaremos las 3 caras (U, F, R) del cubo externo por ahora.
        
        # Posición de inicio (la esquina de un cubie)
        start_pos = -CUBE_SIZE + GAP 

        # Iteradores para recorrer los stickers en el estado del cubo (back-end)
        # La forma en que se dibujan los cubies debe coincidir con los índices de RubikCube.state
        
        # U-face indices: (y=2) [0..8]
        # F-face indices: (z=2) [0..8]
        # R-face indices: (x=2) [0..8]

        # Contadores de stickers para las 6 caras
        sticker_counters = {'U': 0, 'D': 0, 'F': 0, 'B': 0, 'L': 0, 'R': 0}

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    center_x = x * CUBE_SIZE + x * GAP
                    center_y = y * CUBE_SIZE + y * GAP
                    center_z = z * CUBE_SIZE + z * GAP
                    
                    # Colores por defecto para caras no visibles o internas
                    color_u, color_d, color_f, color_b, color_l, color_r = ('K', 'K', 'K', 'K', 'K', 'K') # 'K' para Negro/Internal

                    # Lógica para determinar el color de las caras visibles (x, y, z = 0, 1, 2)
                    
                    # Cara Superior (U)
                    if y == 2:
                        color_u = rubik.get_sticker_color('U', sticker_counters['U'])
                        sticker_counters['U'] += 1
                        
                    # Cara Inferior (D)
                    if y == 0:
                        # NOTA: Los índices de D deben ir del 0 al 8. La asignación es compleja.
                        color_d = rubik.get_sticker_color('D', 8 - (x + (2-z)*3) ) # Mapeo inverso de indices D
                        sticker_counters['D'] += 1 # Solo para contar que se usaron 9
                        
                    # Cara Frontal (F)
                    if z == 2:
                        color_f = rubik.get_sticker_color('F', x + y * 3) # F: x horizontal, y vertical (de 0 a 8)
                        sticker_counters['F'] += 1

                    # Cara Trasera (B)
                    if z == 0:
                        # NOTA: B es compleja. B[0]=top-right, B[2]=top-left
                        color_b = rubik.get_sticker_color('B', 8 - (x*3 + y) ) # Mapeo inverso de B
                        sticker_counters['B'] += 1
                        
                    # Cara Derecha (R)
                    if x == 2:
                        color_r = rubik.get_sticker_color('R', 2 + 3 * y - 3 * x) # R: Indices 2, 5, 8, ...
                        sticker_counters['R'] += 1
                        
                    # Cara Izquierda (L)
                    if x == 0:
                        # NOTA: L es compleja. L[0]=top-left, L[2]=top-right
                        color_l = rubik.get_sticker_color('L', 8 - (2 * y + 3 * (2 - z)))
                        sticker_counters['L'] += 1
                        
                    
                    # Dibujar el cubie con sus colores (sólo si no es el centro vacío)
                    if not (x == 1 and y == 1 and z == 1):
                        self.draw_cubie(
                            center_x, center_y, center_z, 
                            color_f, color_b, color_u, color_d, color_l, color_r
                        )

        # 3. Dibujar Botones y Texto 2D (Overlay)
        self.draw_2d_overlay()


    # ... Nuevo método para dibujar un solo cubie con 6 caras ...
    def draw_cubie(self, x, y, z, color_f, color_b, color_u, color_d, color_l, color_r):
        """Dibuja un solo cubie con sus 6 caras coloreadas."""
        glPushMatrix()
        glTranslatef(x + CUBE_SIZE/2.0, y + CUBE_SIZE/2.0, z + CUBE_SIZE/2.0)
        s = CUBE_SIZE / 2.0
        
        # Usamos el color GRIS para las caras internas (K: Negro)
        # Añade 'K': (0.1, 0.1, 0.1, 1.0) al diccionario COLOR_MAP en la parte superior del archivo.

        # Vertices para las 6 caras
        vertices = [
            # Front (Z+)
            ('v3f', (s, s, s, -s, s, s, -s, -s, s, s, -s, s)), 
            # Back (Z-)
            ('v3f', (s, s, -s, s, -s, -s, -s, -s, -s, -s, s, -s)), 
            # Up (Y+)
            ('v3f', (s, s, s, s, s, -s, -s, s, -s, -s, s, s)),
            # Down (Y-)
            ('v3f', (s, -s, s, -s, -s, s, -s, -s, -s, s, -s, -s)),
            # Left (X-)
            ('v3f', (-s, s, s, -s, s, -s, -s, -s, -s, -s, -s, s)),
            # Right (X+)
            ('v3f', (s, s, s, s, -s, s, s, -s, -s, s, s, -s)),
        ]
        
        # Colores de las 6 caras
        colors = [color_f, color_b, color_u, color_d, color_l, color_r]
        
        # Dibujar cada cara
        for i in range(6):
            glColor4f(*COLOR_MAP[colors[i]])
            pyglet.graphics.draw(4, GL_QUADS, vertices[i])

        glPopMatrix()

    
    
    

    def draw_2d_overlay(self):
        """Dibuja el texto del botón 'Resolver'."""
        self.set_2d() # Cambia a vista 2D
        pyglet.text.Label(
            'Presiona [R] para Revolver. Presiona [A] para Resolver (A*)',
            x=10, y=self.height - 30, color=(255, 255, 255, 255)
        ).draw()
        self.set_3d() # Vuelve a la vista 3D

    def set_2d(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def set_3d(self):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # --- Interacción ---
    def on_key_press(self, symbol, modifiers):
        global rubik
        if symbol == key.A:
            # Llama al Algoritmo A* del back-end
            if not rubik.is_solved():
                print("--- Iniciando Solución A* ---")
                solution = solve_a_star(rubik)
                if solution:
                    print(f"Solución encontrada en {len(solution)} movimientos: {solution}")
                    # Aquí se agregaría la lógica para ejecutar la secuencia visualmente
                else:
                    print("No se encontró solución.")
            else:
                print("El cubo ya está resuelto.")
        
        elif symbol == key.R:
            # Revuelve (usando la simulación incompleta por ahora)
            print("Cubo revuelto (simulación)!")
            rubik = rubik.apply_move('R')
            self.on_draw() # Redibuja

# --- Ejecución ---
if __name__ == '__main__':
    window = RubikWindow(WINDOW_WIDTH, WINDOW_HEIGHT, resizable=True)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST) # Habilita la profundidad para 3D
    pyglet.app.run()