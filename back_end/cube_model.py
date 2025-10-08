# back_end/cube_model.py
import collections
import heapq 

class RubikCube:
    """
    Representa el estado actual del Cubo de Rubik 3x3x3.
    """
    # [U, D, F, B, L, R]
    # U=Up, D=Down, F=Front, B=Back, L=Left, R=Right
    COLORS = {'U': 'W', 'D': 'Y', 'F': 'G', 'B': 'B', 'L': 'R', 'R': 'O'}
    
    def __init__(self):
        """
        Inicializa el cubo en estado resuelto.
        Cada cara ('U', 'D', 'F', 'B', 'L', 'R') contiene una lista de 9 strings (colores).
        """
        # --- CORRECCIÓN CLAVE: Inicializar self.state ---
        self.state = {}
        # ------------------------------------------------
        
        for face, color in self.COLORS.items():
            # [0, 1, 2, 3, 4, 5, 6, 7, 8] donde el centro es el índice 4
            self.state[face] = [color] * 9
    
    def get_sticker_color(self, face, index):
        """Devuelve el color del sticker en una cara e índice específicos."""
        return self.state[face][index]


    def get_state_tuple(self):
        """Devuelve una tupla inmutable para ser usada como clave en diccionarios (Explored Set)."""
        # Aseguramos un orden consistente para la tupla
        return tuple(tuple(self.state[face]) for face in sorted(self.state.keys()))

    def is_solved(self):
        """Verifica si el cubo está resuelto."""
        for face, color in self.COLORS.items():
            if self.state[face] != [color] * 9:
                return False
        return True

    def _rotate_face(self, face_stickers, direction='clockwise'):
        """
        Rota los 8 stickers externos de una cara (2D).
        
        Args:
            face_stickers (list): Lista de 9 stickers de una cara.
            direction (str): 'clockwise' o 'counter_clockwise'.
        
        Returns:
            list: Los 9 stickers con la rotación aplicada.
        """
        new_face = face_stickers[:] # Copia profunda
        
        # Mapeo de rotación: (0, 1, 2, 3, 4, 5, 6, 7, 8) -> (2, 5, 8, 1, 4, 7, 0, 3, 6) para clockwise
        if direction == 'clockwise':
            order = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        else: # counter_clockwise
            order = [2, 5, 8, 1, 4, 7, 0, 3, 6]

        return [face_stickers[i] for i in order]


    # back_end/cube_model.py (Dentro de la clase RubikCube)

# ... (Todo el código anterior de __init__, get_state_tuple, etc. permanece) ...

    # 2. Operadores (Movimientos)
    def apply_move(self, move):
        """
        Aplica un movimiento al cubo y devuelve un *nuevo* objeto RubikCube
        con el estado actualizado.
        """
        new_cube = RubikCube()
        # Copia el estado actual al nuevo cubo (para no modificar el original)
        new_cube.state = {k: list(v) for k, v in self.state.items()} 

        # Diccionario para facilitar las rotaciones de bordes
        # Guarda una lista de tuplas: (cara_origen, indices_origen, cara_destino, indices_destino)
        
        # -----------------------------------------------------------------------------------
        # | U, D, F, B, L, R | Rotación R (Derecha, sentido horario)                        |
        # -----------------------------------------------------------------------------------
        if move == 'R':
            face = 'R'
            
            # 1. Rotación Interna de la Cara R
            new_cube.state[face] = self._rotate_face(new_cube.state[face], 'clockwise')
            
            # 2. Rotación de los 4 Bordes Circundantes
            # La forma más fácil de gestionar esto es almacenar los stickers
            # que se van a mover antes de modificarlos.

            # a) Almacenamos las 4 columnas que se van a rotar:
            U_col = [self.state['U'][2], self.state['U'][5], self.state['U'][8]]
            F_col = [self.state['F'][2], self.state['F'][5], self.state['F'][8]]
            D_col = [self.state['D'][2], self.state['D'][5], self.state['D'][8]]
            B_col = [self.state['B'][6], self.state['B'][3], self.state['B'][0]] # ¡Índices inversos!

            # b) Aplicamos la transferencia (U -> F -> D -> B -> U)
            
            # U -> F (F toma los stickers de U)
            new_cube.state['F'][2], new_cube.state['F'][5], new_cube.state['F'][8] = U_col
            
            # F -> D (D toma los stickers de F)
            new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = F_col
            
            # D -> B (B toma los stickers de D, PERO INVERTIDOS)
            # Los índices de B son inversos: [6, 3, 0]
            new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = D_col
            
            # B -> U (U toma los stickers de B, QUE YA ESTÁN INVERTIDOS EN B_col)
            # B_col es [B6, B3, B0], que es el orden correcto para [U2, U5, U8]
            new_cube.state['U'][2], new_cube.state['U'][5], new_cube.state['U'][8] = B_col
        
        # -----------------------------------------------------------------------------------
        # | Implementación de R' (Derecha, sentido anti-horario)                             |
        # -----------------------------------------------------------------------------------
        elif move == "R'":
            face = 'R'
            
            # 1. Rotación Interna de la Cara R
            new_cube.state[face] = self._rotate_face(new_cube.state[face], 'counter_clockwise')
            
            # 2. Rotación de los 4 Bordes (En el sentido inverso: U -> B -> D -> F -> U)
            
            # a) Almacenamos las 4 columnas:
            U_col = [self.state['U'][2], self.state['U'][5], self.state['U'][8]]
            F_col = [self.state['F'][2], self.state['F'][5], self.state['F'][8]]
            D_col = [self.state['D'][2], self.state['D'][5], self.state['D'][8]]
            B_col = [self.state['B'][6], self.state['B'][3], self.state['B'][0]]

            # b) Aplicamos la transferencia (U -> B -> D -> F -> U)
            
            # U -> B (B toma los stickers de U, PERO INVERTIDOS)
            new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = U_col
            
            # B -> D (D toma los stickers de B, QUE YA ESTÁN INVERTIDOS EN B_col)
            new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = B_col
            
            # D -> F (F toma los stickers de D)
            new_cube.state['F'][2], new_cube.state['F'][5], new_cube.state['F'][8] = D_col
            
            # F -> U (U toma los stickers de F)
            new_cube.state['U'][2], new_cube.state['U'][5], new_cube.state['U'][8] = F_col
            
        # --- PENDIENTE: Añadir lógica para 'U', "U'", 'F', "F'", etc. ---
        
        return new_cube

    def __str__(self):
        """
        Representación en texto del cubo para debugging.
        Muestra una vista desplegada (Cross-shaped).
        """
        def format_row(face, indices):
            return " ".join(self.state.get(face, [' ']*9)[i] for i in indices)
            
        output = []
        # Cara U (Up) - Arriba
        output.append("        " + format_row('U', [0, 1, 2]))
        output.append("        " + format_row('U', [3, 4, 5]))
        output.append("        " + format_row('U', [6, 7, 8]))
        
        # Caras L, F, R, B - Centro (Indices de las bandas)
        for i in range(0, 9, 3):
            row = []
            # L
            row.append(format_row('L', [i, i + 1, i + 2]))
            # F
            row.append(format_row('F', [i, i + 1, i + 2]))
            # R
            row.append(format_row('R', [i, i + 1, i + 2]))
            # B
            row.append(format_row('B', [i, i + 1, i + 2]))
            output.append("  ".join(row))

        # Cara D (Down) - Abajo
        output.append("        " + format_row('D', [0, 1, 2]))
        output.append("        " + format_row('D', [3, 4, 5]))
        output.append("        " + format_row('D', [6, 7, 8]))
        
        return "\n".join(output)


# --- Estructura para el Algoritmo A* (Sin Cambios) ---

class Node:
    """Nodo del árbol de búsqueda para A*."""
    def __init__(self, cube_state, parent=None, action=None, cost=0, heuristic=0):
        self.cube_state = cube_state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic
        self.f_score = self.cost + self.heuristic

    def __lt__(self, other):
        return self.f_score < other.f_score

def heuristic(cube: RubikCube):
    """
    Heurística (h(n)): Estima el costo restante para resolver el cubo.
    """
    incorrect_stickers = 0
    # Recorrer las 6 caras y contar los stickers que no tienen el color central
    for face, color in cube.COLORS.items():
        # El centro (índice 4) es el color base
        expected_color = cube.state[face][4]
        for sticker_color in cube.state[face]:
            if sticker_color != expected_color:
                incorrect_stickers += 1
    # Escala para ser admisible (aproximada para una heurística débil)
    return incorrect_stickers // 8

def solve_a_star(start_cube: RubikCube):
    """
    Algoritmo A* para encontrar la solución óptima.
    """
    # ... (El cuerpo de la función solve_a_star permanece igual) ...
    start_node = Node(start_cube, cost=0, heuristic=heuristic(start_cube))
    frontier = [start_node]
    explored = {start_cube.get_state_tuple(): start_node.cost}
    MOVES = ['R'] # Mantenemos solo 'R' para prueba inicial

    while frontier:
        current_node = heapq.heappop(frontier)
        current_cube = current_node.cube_state

        if current_cube.is_solved():
            path = []
            while current_node.action is not None:
                path.append(current_node.action)
                current_node = current_node.parent
            return path[::-1]

        # Evitamos la explosión del espacio de estados si aún no tenemos todos los movimientos
        if current_node.cost >= 20: 
             continue 

        for move in MOVES:
            new_cube = current_cube.apply_move(move)
            new_state_tuple = new_cube.get_state_tuple()
            new_cost = current_node.cost + 1

            if new_state_tuple in explored and new_cost >= explored[new_state_tuple]:
                continue
                
            new_node = Node(
                new_cube, 
                parent=current_node, 
                action=move, 
                cost=new_cost,
                heuristic=heuristic(new_cube)
            )
            
            explored[new_state_tuple] = new_cost
            heapq.heappush(frontier, new_node)

    return None

if __name__ == '__main__':
    # --- PRUEBA MEJORADA ---
    
    # 1. Cubo Resuelto
    solved_cube = RubikCube()
    print("--- 1. Estado Resuelto (Initial) ---")
    print(solved_cube)
    print(f"¿Resuelto? {solved_cube.is_solved()}") 

      # 2. Aplicar el movimiento real 'R'
    scrambled_cube = solved_cube.apply_move('R')
    
    print("\n--- 2. Estado Revuelto (Movimiento 'R' REAL) ---")
    print(scrambled_cube)
    print(f"Heurística (h(n)): {heuristic(scrambled_cube)}")

    # 3. Prueba de Búsqueda A*
    print("\n--- 3. Búsqueda A* para resolver R' ---")
    # solve_a_star usa MOVES = ['R', "R'", ...]
    solution = solve_a_star(scrambled_cube)
    
    if solution:
        print(f"Solución Encontrada: {solution}")
        print(f"Longitud: {len(solution)}") # Debería ser 1: ['R'']
    else:
        print("Búsqueda A* no encontró solución. Revise la lógica del movimiento.")