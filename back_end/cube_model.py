import os
import math
import copy
import numpy as np
import heapq 
import time # NUEVO: Para medir el tiempo de ejecución
from typing import Optional, Union # NUEVO: Para type hints compatibles con Python 3.9
import sys
from .precomp_cornerslt import get_c8_perm_index
""" GLOBAL DEFINITIONS"""

PDB_C8_TABLE = None
PDB_SIZE_C8_PERM = 40320  # 8! Only permutation. No twist.
PDB_FILENAME_C8 = os.path.join('..', 'assets', 'pdb_c8_perm.npy')

FACE_COLORS = {'U': 'W', 'D': 'Y', 'F': 'G', 'B': 'B', 'L': 'R', 'R': 'O'}

"""
    Corner Index	Slot Name	Piece Colors (Solved)
        0	            UFR	            U, F, R
        1	            UFL	            U, F, L
        2	            UBL	            U, B, L
        3	            UBR	            U, B, R
        4	            DFR	            D, F, R
        5	            DFL	            D, F, L
        6	            DBL	            D, B, L
        7	            DBR	            D, B, R
 
 """
CORNER_COLOR_ID = {
    frozenset(['W', 'G', 'O']): 0, 
    frozenset(['W', 'G', 'R']): 1, 
    frozenset(['W', 'B', 'R']): 2, 
    frozenset(['W', 'B', 'O']): 3, 
    frozenset(['Y', 'G', 'O']): 4, 
    frozenset(['Y', 'G', 'R']): 5,    
    frozenset(['Y', 'B', 'R']): 6, 
    frozenset(['Y', 'B', 'O']): 7
}

CORNER_STICKER_MAP = [
    (('U', 8), ('F', 2), ('R', 0)),  # UFR
    (('U', 6), ('F', 0), ('L', 2)),  # UFL
    (('U', 0), ('B', 2), ('L', 0)),  # UBL
    (('U', 2), ('B', 0), ('R', 2)),  # UBR
    (('D', 2), ('F', 8), ('R', 6)),  # DFR
    (('D', 0), ('F', 6), ('L', 8)),  # DFL
    (('D', 6), ('B', 8), ('L', 6)),  # DBL
    (('D', 8), ('B', 6), ('R', 8)),  # DBR
]


class RubikCube:
    """
    Representa el estado actual del Cubo de Rubik 3x3x3.
    """
    # [U, D, F, B, L, R]

    COLORS = FACE_COLORS
    
    def __init__(self):
        """Inicializa el cubo en estado resuelto."""
        self.state = {
            face: [color] * 9 
            for face, color in self.COLORS.items()
        }
    
    def get_sticker_color(self, face, index):
        """Devuelve el color del sticker en una cara e índice específicos."""
        if face not in self.state:
            return 'X'
        return self.state[face][index]

    def get_state_tuple(self):
        """Devuelve una tupla inmutable para ser usada como clave."""
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
        """
        if direction == 'clockwise':
            # Mapeo: New[i] gets Old[order[i]]
            order = [6, 3, 0, 7, 4, 1, 8, 5, 2] 
        else: # counter_clockwise
            order = [2, 5, 8, 1, 4, 7, 0, 3, 6]

        return [face_stickers[i] for i in order]
    
    
    def __eq__(self, other):
        if not isinstance(other, RubikCube):
            return False
        return self.state == other.state

    def __hash__(self):
        return hash(self.get_state_tuple())

    def _debug_print_F_move(self, move, old_state, new_state):
        """Imprime el estado completo ANTES y DESPUÉS de F/F'."""
        # Se imprime a stderr para evitar conflictos con stdout si el visualizador lo usa.
        print("-" * 60, file=sys.stderr)
        print(f"DEBUG: ESTADO ANTES DEL MOVIMIENTO {move}", file=sys.stderr)
        temp_cube_old = RubikCube()
        temp_cube_old.state = old_state
        print(str(temp_cube_old), file=sys.stderr)
        
        print("\nDEBUG: ESTADO DESPUÉS DEL MOVIMIENTO", file=sys.stderr)
        temp_cube_new = RubikCube()
        temp_cube_new.state = new_state
        print(str(temp_cube_new), file=sys.stderr)
        print("-" * 60, file=sys.stderr) 



    def apply_move(self, move):
        """Aplica un movimiento al cubo."""
        new_cube = RubikCube()
        #new_cube.state = {k: list(v) for k, v in self.state.items()} 
        new_cube.state = copy.deepcopy(self.state) 

        old_state_for_debug = None
        if move in ['F', "F'"]:
             old_state_for_debug = {k: list(v) for k, v in self.state.items()}


        # ---------------------------------------------------------------------
        # R / R' logic (omitted for brevity)
        # ---------------------------------------------------------------------
        if move == 'R' or move == "R'":
            face = 'R'
            clockwise = (move == 'R')

            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            U_col = [self.state['U'][2], self.state['U'][5], self.state['U'][8]]
            F_col = [self.state['F'][2], self.state['F'][5], self.state['F'][8]]
            D_col = [self.state['D'][2], self.state['D'][5], self.state['D'][8]]
            B_col = [self.state['B'][6], self.state['B'][3], self.state['B'][0]]

            if clockwise: # R: U -> F -> D -> B -> U
                new_cube.state['F'][2], new_cube.state['F'][5], new_cube.state['F'][8] = U_col
                new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = F_col
                new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = D_col
                new_cube.state['U'][2], new_cube.state['U'][5], new_cube.state['U'][8] = B_col
                
            else: # R': U -> B -> D -> F -> U

                new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = U_col
                new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = B_col
                new_cube.state['F'][2], new_cube.state['F'][5], new_cube.state['F'][8] = D_col
                new_cube.state['U'][2], new_cube.state['U'][5], new_cube.state['U'][8] = F_col
        
        # L / L'
        elif move == 'L' or move == "L'":
            face = 'L'
            clockwise = (move == 'L')

            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            U_col = [self.state['U'][0], self.state['U'][3], self.state['U'][6]]
            F_col = [self.state['F'][0], self.state['F'][3], self.state['F'][6]]
            D_col = [self.state['D'][0], self.state['D'][3], self.state['D'][6]]
            B_col = [self.state['B'][8], self.state['B'][5], self.state['B'][2]] 


            if clockwise:
                new_cube.state['B'][8], new_cube.state['B'][5], new_cube.state['B'][2] = U_col
                new_cube.state['D'][0], new_cube.state['D'][3], new_cube.state['D'][6] = B_col
                new_cube.state['F'][0], new_cube.state['F'][3], new_cube.state['F'][6] = D_col
                new_cube.state['U'][0], new_cube.state['U'][3], new_cube.state['U'][6] = F_col

            else:
                new_cube.state['F'][0], new_cube.state['F'][3], new_cube.state['F'][6] = U_col
                new_cube.state['D'][0], new_cube.state['D'][3], new_cube.state['D'][6] = F_col
                new_cube.state['B'][8], new_cube.state['B'][5], new_cube.state['B'][2] = D_col
                new_cube.state['U'][0], new_cube.state['U'][3], new_cube.state['U'][6] = B_col

        
        # D / D'
        elif move == 'D' or move == "D'":
            face = 'D'
            clockwise = (move == 'D')

            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            F_row = [self.state['F'][6], self.state['F'][7], self.state['F'][8]]
            R_row = [self.state['R'][6], self.state['R'][7], self.state['R'][8]]
            B_row = [self.state['B'][6], self.state['B'][7], self.state['B'][8]]
            L_row = [self.state['L'][6], self.state['L'][7], self.state['L'][8]]

            if clockwise:
                new_cube.state['R'][6], new_cube.state['R'][7], new_cube.state['R'][8] = F_row
                new_cube.state['B'][6], new_cube.state['B'][7], new_cube.state['B'][8] = R_row
                new_cube.state['L'][6], new_cube.state['L'][7], new_cube.state['L'][8] = B_row
                new_cube.state['F'][6], new_cube.state['F'][7], new_cube.state['F'][8] = L_row
            
            else:
                new_cube.state['L'][6], new_cube.state['L'][7], new_cube.state['L'][8] = F_row
                new_cube.state['B'][6], new_cube.state['B'][7], new_cube.state['B'][8] = L_row
                new_cube.state['R'][6], new_cube.state['R'][7], new_cube.state['R'][8] = B_row
                new_cube.state['F'][6], new_cube.state['F'][7], new_cube.state['F'][8] = R_row


        # U / U'
        elif move == 'U' or move == "U'":
            face = 'U'
            clockwise = (move == 'U')

            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            F_row = [self.state['F'][0], self.state['F'][1], self.state['F'][2]]
            R_row = [self.state['R'][0], self.state['R'][1], self.state['R'][2]]
            B_row = [self.state['B'][0], self.state['B'][1], self.state['B'][2]]
            L_row = [self.state['L'][0], self.state['L'][1], self.state['L'][2]]

            if clockwise:
                new_cube.state['R'][0], new_cube.state['R'][1], new_cube.state['R'][2] = F_row
                new_cube.state['B'][0], new_cube.state['B'][1], new_cube.state['B'][2] = R_row
                new_cube.state['L'][0], new_cube.state['L'][1], new_cube.state['L'][2] = B_row
                new_cube.state['F'][0], new_cube.state['F'][1], new_cube.state['F'][2] = L_row

            else:
                new_cube.state['L'][0], new_cube.state['L'][1], new_cube.state['L'][2] = F_row
                new_cube.state['B'][0], new_cube.state['B'][1], new_cube.state['B'][2] = L_row
                new_cube.state['R'][0], new_cube.state['R'][1], new_cube.state['R'][2] = B_row
                new_cube.state['F'][0], new_cube.state['F'][1], new_cube.state['F'][2] = R_row

        # ---------------------------------------------------------------------
        # F / F' (Frontal) - Lógica de inversión con Debugging
        # ---------------------------------------------------------------------
        elif move == 'F' or move == "F'":
            face = 'F'
            clockwise = (move == 'F')

            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            U_row = [self.state['U'][6], self.state['U'][7], self.state['U'][8]] 
            R_col = [self.state['R'][0], self.state['R'][3], self.state['R'][6]] 
            D_row = [self.state['D'][0], self.state['D'][1], self.state['D'][2]] 
            L_col = [self.state['L'][2], self.state['L'][5], self.state['L'][8]] 

            if clockwise: # F: U -> R -> D -> L -> U (TODAS INVERTIDAS)
                new_cube.state['R'][0], new_cube.state['R'][3], new_cube.state['R'][6] = U_row[::-1]
                new_cube.state['D'][0], new_cube.state['D'][1], new_cube.state['D'][2] = R_col
                new_cube.state['L'][8], new_cube.state['L'][5], new_cube.state['L'][2] = D_row[::-1]
                new_cube.state['U'][6], new_cube.state['U'][7], new_cube.state['U'][8] = L_col[::-1]

            else: # F' (Anti-Clockwise): U -> L -> D -> R -> U 
                new_cube.state['L'][2], new_cube.state['L'][5], new_cube.state['L'][8] = U_row
                new_cube.state['D'][0], new_cube.state['D'][1], new_cube.state['D'][2] = L_col[::-1]
                new_cube.state['R'][6], new_cube.state['R'][3], new_cube.state['R'][0] = D_row[::-1]
                new_cube.state['U'][6], new_cube.state['U'][7], new_cube.state['U'][8] = R_col


            self._debug_print_F_move(move, old_state_for_debug, new_cube.state)
                
        # B / B' (Trasera)
        elif move == 'B' or move == "B'":
            face = 'B'
            clockwise = (move == 'B') 
            
            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            U_row = [self.state['U'][0], self.state['U'][1], self.state['U'][2]]
            L_col = [self.state['L'][0], self.state['L'][3], self.state['L'][6]]
            D_row = [self.state['D'][6], self.state['D'][7], self.state['D'][8]]
            R_col = [self.state['R'][2], self.state['R'][5], self.state['R'][8]]


            if clockwise: # B: U -> L -> D -> R -> U
                new_cube.state['L'][0], new_cube.state['L'][3], new_cube.state['L'][6] = U_row[::-1]
                new_cube.state['D'][8], new_cube.state['D'][7], new_cube.state['D'][6] = L_col[::-1]
                new_cube.state['R'][2], new_cube.state['R'][5], new_cube.state['R'][8] = D_row[::-1]
                new_cube.state['U'][0], new_cube.state['U'][1], new_cube.state['U'][2] = R_col

            else: # B': U -> R -> D -> L -> U
                new_cube.state['R'][2], new_cube.state['R'][5], new_cube.state['R'][8] = U_row                
                new_cube.state['D'][6], new_cube.state['D'][7], new_cube.state['D'][8] = R_col[::-1]
                new_cube.state['L'][0], new_cube.state['L'][3], new_cube.state['L'][6] = D_row
                new_cube.state['U'][0], new_cube.state['U'][1], new_cube.state['U'][2] = L_col[::-1]
        
        return new_cube

    def __str__(self):
        """
        Representación en texto del cubo en formato de red 2D (el formato solicitado).
        """
        def format_face_row(face, indices, separator=' '):
            return separator.join(self.state.get(face, [' ']*9)[i] for i in indices)
            
        output = []

        # 1. Cara U (Up)
        for i in [0, 3, 6]:
            output.append("        " + format_face_row('U', [i, i + 1, i + 2]))
        
        # 2. Caras L, F, R, B
        for i in [0, 3, 6]:
            L_row = format_face_row('L', [i, i + 1, i + 2])
            F_row = format_face_row('F', [i, i + 1, i + 2])
            R_row = format_face_row('R', [i, i + 1, i + 2])
            B_row = format_face_row('B', [i, i + 1, i + 2])
            output.append(f"{L_row}  {F_row}  {R_row}  {B_row}")

        # 3. Cara D (Down)
        for i in [0, 3, 6]:
            output.append("        " + format_face_row('D', [i, i + 1, i + 2]))
        
        return "\n".join(output)


# --- Estructura para el Algoritmo A* ---

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
    """Heurística simple."""
    incorrect_stickers = 0
    for face, color in cube.COLORS.items():
        expected_color = cube.state[face][4]
        for sticker_color in cube.state[face]:
            if sticker_color != expected_color:
                incorrect_stickers += 1
    return incorrect_stickers // 8

# AHORA DEVUELVE 3 VALORES (solución, nodos, tiempo) y usa type hints compatibles.
def solve_a_star(start_cube: RubikCube, max_depth: int = 30) -> tuple[Optional[list[str]], int, float]: 
    """Algoritmo A*."""
    start_time = time.time() # Iniciar el temporizador

    start_node = Node(start_cube, cost=0, heuristic=heuristic(start_cube))
    frontier = [start_node]
    explored = {start_cube.get_state_tuple(): start_node.cost}
    MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"] 

    expanded_nodes = 0

    while frontier:
        current_node = heapq.heappop(frontier)
        current_cube = current_node.cube_state
        expanded_nodes += 1

        if current_cube.is_solved():
            end_time = time.time() # Detener el temporizador
            path = []
            while current_node.action is not None:
                path.append(current_node.action)
                current_node = current_node.parent
            return path[::-1], expanded_nodes, end_time - start_time # Retornar (solución, nodos, tiempo)

        if expanded_nodes >= 200000: 
             break 

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

    end_time = time.time() # Detener el temporizador si no encuentra solución
    return None, expanded_nodes, end_time - start_time # Retornar (None, nodos, tiempo)


# --- Algoritmo IDA* (Iterative Deepening A*) ---
def _opposite_move(move: str) -> str:
    """Devuelve el movimiento inverso para evitar deshacer inmediatamente."""
    if move.endswith("'"):
        return move[:-1]
    else:
        return move + "'"

# AHORA DEVUELVE 3 VALORES (solución, nodos, tiempo) y usa type hints compatibles.
def solve_ida_star(start_cube: RubikCube, max_depth: int = 30) -> tuple[Optional[list[str]], int, float]:
    """
    Resuelve usando IDA* (iterative deepening A*).
    Retorna (secuencia_de_movimientos, nodos_expandidos, tiempo_en_segundos). Si no encuentra solución dentro de max_depth, devuelve (None, nodos_expandidos, tiempo).
    """
    start_time = time.time() # Iniciar el temporizador
    MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]
    #threshold = heuristic(start_cube)
    #threshold = corner_twist_heuristic(start_cube)

    start_h = heuristic(start_cube)
    threshold = start_h
    expanded = 0

    # FIX CRÍTICO: Uso de Union y Optional para compatibilidad con Python 3.9
    def search(cube: RubikCube, g: int, threshold: int, last_move: Optional[str], path: list[str]) -> tuple[Union[int, str], Optional[list[str]]]:
        nonlocal expanded

        #h = heuristic(cube)
        #h = corner_twist_heuristic(cube)
        h = heuristic(cube)

        f = g + h
        if f > threshold:
            return f, None
        if cube.is_solved():
            return "FOUND", path

        if g >= max_depth:
            # Evita explorar más allá del límite duro
            return float('inf'), None

        min_next = float('inf')
        expanded += 1

        for m in MOVES:
            # Evitar deshacer el último movimiento de inmediato
            if last_move and m == _opposite_move(last_move):
                continue

            child = cube.apply_move(m)
            # IDA* clásico en profundidad primero
            res, sol = search(child, g + 1, threshold, m, path + [m])
            if res == "FOUND":
                return "FOUND", sol
            if isinstance(res, (int, float)) and res < min_next:
                min_next = res

        return min_next, None

    # Bucle de profundización por umbral f = g + h
    while True:
        result, sol = search(start_cube, 0, threshold, None, [])
        if result == "FOUND":
            end_time = time.time() # Detener el temporizador
            return sol, expanded, end_time - start_time # Retornar 3 valores
        if isinstance(result, (int, float)) and result == float('inf'):
            end_time = time.time() # Detener el temporizador
            # agotamos sin nuevas fronteras útiles
            return None, expanded, end_time - start_time # Retornar 3 valores
        threshold = int(result)  # elevar umbral al menor f que superó el threshold
        if threshold > max_depth:
            end_time = time.time() # Detener el temporizador
            return None, expanded, end_time - start_time # Retornar 3 valores
