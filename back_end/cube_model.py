import os
import math
import numpy as np
import heapq 
import sys
from .precomp_cornerslt import get_c8_perm_index
""" GLOBAL DEFINITIONS"""

PDB_C8_TABLE = None
PDB_SIZE_C8_PERM = 40320  # 8! Only permutation. No twist.
PDB_FILENAME_C8 = os.path.join('..', 'assets', 'pdb_c8_perm.npy')

FACE_COLORS = {'U': 'W', 'D': 'Y', 'F': 'B', 'B': 'G', 'L': 'O', 'R': 'R'}

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
    frozenset(['W', 'B', 'R']): 0, 
    frozenset(['W', 'B', 'O']): 1, 
    frozenset(['W', 'G', 'O']): 2, 
    frozenset(['W', 'G', 'R']): 3, 
    frozenset(['Y', 'B', 'R']): 4, 
    frozenset(['Y', 'B', 'O']): 5,    
    frozenset(['Y', 'G', 'O']): 6, 
    frozenset(['Y', 'G', 'R']): 7
}

CORNER_STICKER_MAP = [
    # (U-sticker, F-sticker, R-sticker) - Solved Orientation 0
    (('U', 8), ('F', 2), ('R', 0)), # UFR
    (('U', 6), ('F', 0), ('L', 2)), # UFL
    (('U', 0), ('B', 2), ('L', 0)), # UBL
    (('U', 2), ('B', 0), ('R', 2)), # UBR
    # (D-sticker, F-sticker, R-sticker) - Solved Orientation 0
    (('D', 2), ('F', 8), ('R', 6)), # DFR
    (('D', 0), ('F', 6), ('L', 8)), # DFL
    (('D', 6), ('B', 8), ('L', 6)), # DBL
    (('D', 8), ('B', 6), ('R', 8)), # DBR
]


class RubikCube:
    """
    Representa el estado actual del Cubo de Rubik 3x3x3.
    """
    # [U, D, F, B, L, R]
    COLORS = {'U': 'W', 'D': 'Y', 'F': 'B', 'B': 'G', 'L': 'O', 'R': 'R'}
    
    def __init__(self):
        """Inicializa el cubo en estado resuelto."""
        self.state = {}
        
        for face, color in self.COLORS.items():
            # [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.state[face] = [color] * 9
    
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
        new_stickers = [0] * 9
        if direction == 'clockwise':
            # Mapeo: New[i] gets Old[order[i]]
            order = [6, 3, 0, 7, 4, 1, 8, 5, 2] 
        else: # counter_clockwise
            order = [2, 5, 8, 1, 4, 7, 0, 3, 6]

        for new_idx, old_idx in enumerate(order):
            new_stickers[new_idx] = face_stickers[old_idx]

        #return [face_stickers[i] for i in order]
        return new_stickers
    
 
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
        new_cube.state = {k: list(v) for k, v in self.state.items()} 
        
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
            B_col_rev = [self.state['B'][6], self.state['B'][3], self.state['B'][0]]
            if clockwise: # R: U -> F -> D -> B -> U
                new_cube.state['F'][2], new_cube.state['F'][5], new_cube.state['F'][8] = U_col
                new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = F_col
                new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = D_col[::-1]
                new_cube.state['U'][2], new_cube.state['U'][5], new_cube.state['U'][8] = B_col_rev[::-1]
            else: # R': U -> B -> D -> F -> U
                new_cube.state['B'][6], new_cube.state['B'][3], new_cube.state['B'][0] = U_col[::-1]
                new_cube.state['D'][2], new_cube.state['D'][5], new_cube.state['D'][8] = B_col_rev[::-1]
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
            B_col_rev = [self.state['B'][8], self.state['B'][5], self.state['B'][2]] 
            col_idx = [0, 3, 6]
            B_idx_rev = [8, 5, 2] 
            if clockwise:
                for i, val in zip(col_idx, U_col): new_cube.state['F'][i] = val
                for i, val in zip(col_idx, F_col): new_cube.state['D'][i] = val
                for i, val in zip(B_idx_rev, D_col[::-1]): new_cube.state['B'][i] = val
                for i, val in zip(col_idx, B_col_rev[::-1]): new_cube.state['U'][i] = val
            else:
                for i, val in zip(B_idx_rev, U_col[::-1]): new_cube.state['B'][i] = val
                for i, val in zip(col_idx, B_col_rev[::-1]): new_cube.state['D'][i] = val
                for i, val in zip(col_idx, D_col): new_cube.state['F'][i] = val
                for i, val in zip(col_idx, F_col): new_cube.state['U'][i] = val
        
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
            row_idx = [6, 7, 8]
            if clockwise:
                for i, val in zip(row_idx, F_row): new_cube.state['R'][i] = val
                for i, val in zip(row_idx, R_row): new_cube.state['B'][i] = val
                for i, val in zip(row_idx, B_row): new_cube.state['L'][i] = val
                for i, val in zip(row_idx, L_row): new_cube.state['F'][i] = val
            else:
                for i, val in zip(row_idx, F_row): new_cube.state['L'][i] = val
                for i, val in zip(row_idx, L_row): new_cube.state['B'][i] = val
                for i, val in zip(row_idx, B_row): new_cube.state['R'][i] = val
                for i, val in zip(row_idx, R_row): new_cube.state['F'][i] = val

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
            row_idx = [0, 1, 2]
            if clockwise:
                for i, val in zip(row_idx, F_row): new_cube.state['R'][i] = val
                for i, val in zip(row_idx, R_row): new_cube.state['B'][i] = val
                for i, val in zip(row_idx, B_row): new_cube.state['L'][i] = val
                for i, val in zip(row_idx, L_row): new_cube.state['F'][i] = val
            else:
                for i, val in zip(row_idx, F_row): new_cube.state['L'][i] = val
                for i, val in zip(row_idx, L_row): new_cube.state['B'][i] = val
                for i, val in zip(row_idx, B_row): new_cube.state['R'][i] = val
                for i, val in zip(row_idx, R_row): new_cube.state['F'][i] = val
        
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
                new_cube.state['D'][0], new_cube.state['D'][1], new_cube.state['D'][2] = R_col[::-1]
                new_cube.state['L'][2], new_cube.state['L'][5], new_cube.state['L'][8] = D_row[::-1]
                new_cube.state['U'][6], new_cube.state['U'][7], new_cube.state['U'][8] = L_col[::-1]

            else: # F' (Anti-Clockwise): U -> L -> D -> R -> U 
                new_cube.state['L'][2], new_cube.state['L'][5], new_cube.state['L'][8] = U_row[::-1]
                new_cube.state['D'][0], new_cube.state['D'][1], new_cube.state['D'][2] = L_col[::-1]
                new_cube.state['R'][0], new_cube.state['R'][3], new_cube.state['R'][6] = D_row[::-1]
                new_cube.state['U'][6], new_cube.state['U'][7], new_cube.state['U'][8] = R_col[::-1]

            self._debug_print_F_move(move, old_state_for_debug, new_cube.state)
                
        # B / B' (Trasera)
        elif move == 'B' or move == "B'":
            face = 'B'
            clockwise = (move == 'B') 
            
            direction = 'clockwise' if clockwise else 'counter_clockwise'
            new_cube.state[face] = self._rotate_face(new_cube.state[face], direction)

            U_row = [self.state['U'][0], self.state['U'][1], self.state['U'][2]]
            R_col = [self.state['R'][2], self.state['R'][5], self.state['R'][8]]
            D_row = [self.state['D'][6], self.state['D'][7], self.state['D'][8]]
            L_col = [self.state['L'][0], self.state['L'][3], self.state['L'][6]]

            
            U_inv = [U_row[2], U_row[1], U_row[0]]
            R_inv = [R_col[2], R_col[1], R_col[0]]
            D_inv = [D_row[2], D_row[1], D_row[0]]
            L_inv = [L_col[2], L_col[1], L_col[0]]
            
    

            if clockwise: # B: U -> L -> D -> R -> U
                new_cube.state['L'][0], new_cube.state['L'][3], new_cube.state['L'][6] = U_inv
                new_cube.state['D'][6], new_cube.state['D'][7], new_cube.state['D'][8] = L_inv
                new_cube.state['R'][2], new_cube.state['R'][5], new_cube.state['R'][8] = D_inv
                new_cube.state['U'][0], new_cube.state['U'][1], new_cube.state['U'][2] = R_inv

            else: # B': U -> R -> D -> L -> U
                new_cube.state['R'][2], new_cube.state['R'][5], new_cube.state['R'][8] = U_inv
                new_cube.state['D'][6], new_cube.state['D'][7], new_cube.state['D'][8] = R_inv
                new_cube.state['L'][0], new_cube.state['L'][3], new_cube.state['L'][6] = D_inv
                new_cube.state['U'][0], new_cube.state['U'][1], new_cube.state['U'][2] = L_inv
        
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


# --- Estructura para el Algoritmo A* (Omitido por brevedad) ---

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

def corner_twist_heuristic(cube: RubikCube):

    """
        ADMISSIBLE h <= 5.
        Calculates the orientation (twist: 0, 1, or 2) for all 8 corners.

        Orientation is determined by which face color (W/Y) is on the U/D face.
        - 0: The U/D-face sticker has the U/D center color (solved).
        - 1 or 2: The U/D-face sticker has an F/B/L/R center color (twisted).

    """
    
    twist_sum = 0
    # Identify the solved center colors

    U_COLOR = FACE_COLORS['U'] # i.e 'W'
    D_COLOR = FACE_COLORS['D'] # i.e 'Y'
    
    for i, ((face_u_d, idx_u_d), (face_f_b, idx_f_b), (face_l_r, idx_l_r)) in enumerate(CORNER_STICKER_MAP):
        

        # 1. Determine which cubie we are tracking (which 3 colors)
        sticker_color_u_d = cube.get_sticker_color(face_u_d, idx_u_d)
        
        # 2. Check the U/D sticker's color
        if i <= 3: # UPPER corners (UFR, UFL, UBL, UBR)
            solved_color = U_COLOR
        else:      # DOWN corners (DFR, DFL, DBL, DBR)
            solved_color = D_COLOR

        # 3. Calculate Twist
        twist = 0
        if sticker_color_u_d != solved_color:
            # If the U/D face color is incorrect, the corner is twisted (1 or 2).
            # A simpler, admissible approach is to check if the U/D sticker is correct.
            
            # Simplified Twist Calculation (Admissible but not exact twist value)
            # If   U/D color is on F/B side -> twist = 1
            # Else U/D color is on L/R side -> twist = 2
            
            sticker_color_f_b = cube.get_sticker_color(face_f_b, idx_f_b)
            sticker_color_l_r = cube.get_sticker_color(face_l_r, idx_l_r)
            
            if sticker_color_f_b == solved_color:
                # U/D color is on the second face (F/B)
                twist = 1
            elif sticker_color_l_r == solved_color:
                # U/D color is on the third face (L/R)
                twist = 2
            
        twist_sum += twist
    
    # Since a single move can correct the twist of at most 3 units (due to the 0 mod 3 constraint),
    # it follows the minimum number of moves is the total sum divided by 3, rounded up.
    
    if twist_sum == 0:
        return 0
    
    h_twist = math.ceil(twist_sum / 3)
    
    return int(h_twist)

def corner_perm_heuristic(cube: RubikCube):

    global PDB_C8_TABLE
    if PDB_C8_TABLE is None:
        try:
            PDB_C8_TABLE = np.fromfile(PDB_FILENAME_C8, dtype = np.int8)
        except FileNotFoundError:
            PDB_C8_TABLE = np.full(1, 0, dtype=np.int8)


    # --- Corner Cubie Map (Indices 0-7) ---
    # This is how the PDB defines the slots/positions (0-7):
    # U-layer: 0:UFR, 1:UFL, 2:UBL, 3:UBR
    # D-layer: 4:DFR, 5:DFL, 6:DBL, 7:DBR

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

    """
    What this crap means?
    e.j.  R[0] = 4.
    Corner at index 4 (DFR) moves to corner at index 0 (UFR)

    i.e. TABLE[i] = j <-> j cubie goes to i cubie where i, j are corners.

    """

    def cube_corner_perm(cube):
        """
        RubikCube state into the PDB's 8-piece permutation.
    
        """
        perm = [0] * 8
    
        for slot_index, (sticker_ud, sticker_fb, sticker_lr) in enumerate(CORNER_STICKER_MAP):
        
                    # 1. Read the three colors
            colors_list = []
        
            colors_list.append(cube.get_sticker_color(sticker_ud[0], sticker_ud[1]))
            colors_list.append(cube.get_sticker_color(sticker_fb[0], sticker_fb[1]))
            colors_list.append(cube.get_sticker_color(sticker_lr[0], sticker_lr[1]))
        
            # 2. CRITICAL FIX: Filter out invalid/sentinel values
            # Assumes valid colors are strings like 'W', 'Y', 'R', etc.
            valid_colors = {c for c in colors_list if c is not None and c != 'X' and c != ''}
        
            # Check if we failed to read all 3 stickers
            if len(valid_colors) != 3:
            # Handle the error gracefully, maybe print a full debug of the raw colors_list
            # The previous FATAL ERROR print below will already happen, but this is why
                pass
        
            current_colors = frozenset(valid_colors)
        
            # 2. Identify the piece index (j) from its unique color set
            try:
                piece_index = CORNER_COLOR_ID[current_colors]
            except KeyError:
            # ... (Your existing error logging code) ...
                color_str = ", ".join(sorted(list(current_colors)))
            # The current output of colors_list in your original error was only 2 colors: {G, W}
            # This confirms the filtering step is necessary.
                print(f"\nFATAL ERROR: Slot Index {slot_index} (Piece: {slot_index})")
                print(f"Sticker Map: {sticker_ud}, {sticker_fb}, {sticker_lr}")
            # Correcting the output to show the actual problem:
                print(f"Cube Colors Read (Invalid Set): {{{color_str}}}") 
                print(f"Full List Read: {colors_list}") # <-- Use this for further debugging
                print(f"Expected Piece Colors (CORNER_COLOR_ID): {CORNER_COLOR_ID}")
                return None 

            perm[slot_index] = piece_index
        
        return perm
    
    perm_state = cube_corner_perm(cube)

    
    index = get_c8_perm_index(perm_state)
    
    if len(PDB_C8_TABLE) == PDB_SIZE_C8_PERM:
        print("FROM PDB TABLE")
        return int(PDB_C8_TABLE[index])
    else:
        #Shit's corrupted, must failsafe
        return corner_twist_heuristic(cube) 


def solve_a_star(start_cube: RubikCube):
    """Algoritmo A*."""
    start_node = Node(start_cube, cost=0, heuristic=heuristic(start_cube))
    #start_node = Node(start_cube, cost=0, heuristic=corner_twist_heuristic(start_cube))
    #start_node = Node(start_cube, cost=0, heuristic=corner_perm_heuristic(start_cube))
    frontier = [start_node]
    explored = {start_cube.get_state_tuple(): start_node.cost}
    MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"] 

    expanded_nodes = 0

    while frontier:
        current_node = heapq.heappop(frontier)
        current_cube = current_node.cube_state
        expanded_nodes += 1

        if current_cube.is_solved():
            path = []
            while current_node.action is not None:
                path.append(current_node.action)
                current_node = current_node.parent
            return path[::-1], expanded_nodes

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
                #heuristic = corner_twist_heuristic(new_cube)
                #heuristic = corner_perm_heuristic(new_cube)
            )
            
            explored[new_state_tuple] = new_cost
            heapq.heappush(frontier, new_node)

    return None, expanded_nodes

# === Cambios propuestos para integrar IDA* (tecla J) sin quitar A* ===
# 
# Copia/pega estos bloques en tus archivos.
# Mantengo intacto tu A* y la heurística actual; IDA* los reutiliza.

# -----------------------------------------------------------------------------
# 1) back_end/cube_model.py  →  Agregar al final del archivo
# -----------------------------------------------------------------------------

# --- Algoritmo IDA* (Iterative Deepening A*) ---
# Usa la misma heurística() ya definida. Recorre en profundidad con umbral f=g+h
# y aumenta el umbral iterativamente. Incluye un pruning básico para no deshacer
# el último movimiento inmediatamente.

def _opposite_move(move):
    """Devuelve el movimiento inverso (p.ej. R' ↔ R) para evitar backtracking inmediato."""
    return move[:-1] if move.endswith("'") else move + "'"


def solve_ida_star(start_cube, max_depth=30):
    """
    Resuelve usando IDA* (iterative deepening A*).
    Retorna (secuencia_de_movimientos, nodos_expandidos).
    Si no encuentra solución dentro de max_depth, devuelve (None, nodos_expandidos).
    """
    MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]
    threshold = heuristic(start_cube)
    #threshold = corner_twist_heuristic(start_cube)
    #threshold = corner_perm_heuristic(start_cube)
    expanded = 0

    def search(cube, g, threshold, last_move, path):
        nonlocal expanded
        h = heuristic(cube)
        #h = corner_twist_heuristic(cube)
        #h = corner_perm_heuristic(cube)
        f = g + h
        if f > threshold:
            return f, None
        if cube.is_solved():
            return "FOUND", path
        if g >= max_depth:
            return float('inf'), None

        min_next = float('inf')
        expanded += 1

        for m in MOVES:
            # Evitar deshacer el último movimiento
            if last_move and m == _opposite_move(last_move):
                continue

            child = cube.apply_move(m)
            res, sol = search(child, g + 1, threshold, m, path + [m])
            if res == "FOUND":
                return "FOUND", sol
            if isinstance(res, (int, float)) and res < min_next:
                min_next = res

        return min_next, None

    while True:
        result, sol = search(start_cube, 0, threshold, None, [])
        if result == "FOUND":
            return sol, expanded
        if isinstance(result, (int, float)) and result == float('inf'):
            return None, expanded
        threshold = int(result)
        if threshold > max_depth:
            return None, expanded


