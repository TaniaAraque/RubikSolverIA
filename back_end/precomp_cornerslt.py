import os
import numpy as np
from collections import deque

""" GLOBAL DEFINITIONS """

PDB_SIZE_C8_PERM = 40320  # 8! Only permutation. No twist.
PDB_FILENAME_C8 = os.path.join('..', 'assets', 'pdb_c8_perm.npy')

MOVES = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]


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
PERMUTATION_MAPS = {
    'R': [4, 1, 0, 3, 6, 5, 2, 7], "R'": [2, 1, 6, 3, 0, 5, 4, 7],
    'L': [0, 3, 2, 7, 4, 1, 6, 5], "L'": [0, 5, 2, 1, 4, 7, 6, 3],
    'U': [3, 0, 1, 2, 4, 5, 6, 7], "U'": [1, 2, 3, 0, 4, 5, 6, 7],
    'D': [0, 1, 2, 3, 7, 4, 5, 6], "D'": [0, 1, 2, 3, 5, 6, 7, 4],
    'F': [1, 5, 2, 3, 0, 4, 6, 7], "F'": [4, 0, 2, 3, 5, 1, 6, 7],
    'B': [0, 1, 7, 6, 4, 5, 3, 2], "B'": [0, 1, 2, 6, 4, 5, 7, 3]
}


def get_c8_perm_index(perm):
    """
    Correct Factoradic index (Permutation Rank) that yields 0 for the solved state.
    It counts how many *available* pieces are smaller than the piece at slot perm[i].
    Range: 0 to 40319 (8!)
    """
    index = 0
    # Factorials (0! to 7!) for weights: 1, 1, 2, 6, 24, 120, 720, 5040
    factorials = [1, 1, 2, 6, 24, 120, 720, 5040] 
    
    # Track which piece IDs (0-7) have been used so far
    available_pieces = list(range(8))
    
    for i in range(8): 
        piece_id = perm[i]
        
        # 1. Find the current rank of piece_id among the available_pieces
        # This count is the Factoradic coefficient for this position (c_i).
        rank = available_pieces.index(piece_id)
        
        # 2. The weight for this position is (7 - i)!
        weight_index = 7 - i 
        
        # 3. Add to the index (rank * weight)
        index += rank * factorials[weight_index]
        
        # 4. Remove the piece_id from the list of available pieces
        available_pieces.pop(rank)
            
    return index


def apply_move_to_perm(current_perm, move):
    """Applies a move and returns the new corner permutation state."""
    p_map = PERMUTATION_MAPS[move]
    # The piece that was at old_pos moves to new_pos = p_map[old_pos]
    
    new_perm = [0] * 8
    for i in range(8):
        # The piece 'current_perm[i]' moves to slot 'p_map[i]'
        new_perm[p_map[i]] = current_perm[i]
    return new_perm


def generate_c8_perm_pdb():
    """
    Precomputed 8-Corner Permutation PDB lookup table.
    Executed only if table not present
    """
    
    if os.path.exists(PDB_FILENAME_C8):
        return
        
    # PDB is small enough for simple initialization
    pdb_table = np.full(PDB_SIZE_C8_PERM, -1, dtype=np.int8) 
    
    # SOLVED STATE. IDENTITY PERM.
    solved_perm = [0, 1, 2, 3, 4, 5, 6, 7] 
    start_index = get_c8_perm_index(solved_perm)
    

    #"REVERSE BFS". Compute table.
    pdb_table[start_index] = 0
    queue = deque([solved_perm])
    
    max_depth = 0
    
    
    while queue:
        current_perm = queue.popleft()
        current_index = get_c8_perm_index(current_perm)
        depth = pdb_table[current_index]
        
        if depth > max_depth:
            max_depth = depth
            
        for move in MOVES:
            next_perm = apply_move_to_perm(current_perm, move)
            next_index = get_c8_perm_index(next_perm)
            
            if pdb_table[next_index] == -1:
                pdb_table[next_index] = depth + 1
                queue.append(next_perm)
                
    pdb_table.tofile(PDB_FILENAME_C8)


if __name__ == "__main__":
    generate_c8_perm_pdb()
