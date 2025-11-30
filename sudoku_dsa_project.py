# ---------------- Imports & Title -----------------------------------------------------------
import copy
import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Sudoku Solver — MRV + Forward Checking (with your helpers)")  


# 1. Base puzzle ---------------------------------------------------------------------------
puzzle_default = [
    [5,3,0, 0,7,0, 0,0,0],   
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],
    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],
    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,8,0, 0,7,9],   
]


# 2. Helper functions ------------------------------------------------------------------------
def empty_cells(board):
    """Return list of all empty cells as (row, col) pairs."""
    cells = []
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                cells.append([r, c])
    return cells


def is_valid(board, r, c, val):
    """Check whether `val` can be placed at position (r, c)."""
    
    # Check row
    if val in board[r]:
        return False
    
    # Check column
    for i in range(9):
        if board[i][c] == val:
            return False
    
    # Check 3×3 box
    box_r = 3 * (r // 3)
    box_c = 3 * (c // 3)
    for i in range(box_r, box_r + 3):
        for j in range(box_c, box_c + 3):
            if board[i][j] == val:
                return False
    
    return True


def board_to_key(board):
    """Convert board into a string key for visited-state pruning."""
    key = ""
    for row in board:
        for x in row:
            key += str(x)
    return key


def find_pruning_cell(board):
    """MRV heuristic: return the empty cell with the fewest valid candidates."""
    best_cell = None
    best_count = 10  # larger than max possible candidates
    
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                count = 0
                for k in range(1, 10):
                    if is_valid(board, r, c, k):
                        count += 1
                if count < best_count:
                    best_count = count
                    best_cell = [r, c]
                if best_count == 1:
                    return best_cell
    return best_cell


def possible_values(board, r, c):
    """Return all valid candidate values for cell (r, c)."""
    vals = []
    for v in range(1, 10):
        if is_valid(board, r, c, v):
            vals.append(v)
    return vals


# 3. Forward Checking helper -----------------------------------------------------------------
def forward_check_ok(board, r, c):
    """Check if all neighbors of (r, c) still have at least one valid candidate."""
    
    # Check row and column
    for k in range(9):
        if board[r][k] == 0 and not possible_values(board, r, k):
            return False
        if board[k][c] == 0 and not possible_values(board, k, c):
            return False
    
    # Check 3×3 block
    br = 3 * (r // 3)
    bc = 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == 0 and not possible_values(board, i, j):
                return False
    
    return True


# 4. Stats + Recursive solver ----------------------------------------------------------------
def init_stats():
    """Initialize a dictionary to track stats."""
    return {"dfs_calls": 0, "back_track": 0, "depth_max": 0}


def solve_sudoku(board, visited=None, stats=None, depth=0):
    """Recursive solver using MRV + state pruning + early termination."""
    
    if visited is None:
        visited = set()
    if stats is None:
        stats = init_stats()

    stats["dfs_calls"] += 1
    stats["depth_max"] = max(stats["depth_max"], depth)

    key = board_to_key(board)
    if key in visited:
        return False
    visited.add(key)

    positions = empty_cells(board)
    if not positions:
        return True

    r, c = find_pruning_cell(board)
    vals = possible_values(board, r, c)

    # Early termination if only 1 candidate
    if len(vals) == 1:
        only = vals[0]
        board[r][c] = only
        if solve_sudoku(board, visited, stats, depth + 1):
            return True
        board[r][c] = 0
        stats["back_track"] += 1
        return False

    # Try all candidates normally
    for val in vals:
        board[r][c] = val
        if solve_sudoku(board, visited, stats, depth + 1):
            return True
        board[r][c] = 0
        stats["back_track"] += 1

    return False


# 5. Step generator for MRV + Forward Checking (used by Streamlit) ----------------------------
def solve_with_steps_fc(initial_board):
    """Generator yielding each step (place/backtrack/done) for UI animation."""
    
    b = copy.deepcopy(initial_board)
    stack = []  # (row, col, values, value_index)

    while True:
        empties = empty_cells(b)
        if not empties:
            yield ('done', -1, -1, -1, copy.deepcopy(b))
            return

        r, c = find_pruning_cell(b)
        vals = possible_values(b, r, c)

        # If no values, we must backtrack
        if not vals:
            while stack:
                pr, pc, pvals, idx = stack.pop()
                b[pr][pc] = 0
                yield ('backtrack', pr, pc, -1, copy.deepcopy(b))

                idx += 1
                while idx < len(pvals):
                    v = pvals[idx]
                    if is_valid(b, pr, pc, v):
                        b[pr][pc] = v
                        if forward_check_ok(b, pr, pc):
                            yield ('place', pr, pc, v, copy.deepcopy(b))
                            stack.append((pr, pc, pvals, idx))
                            break
                        b[pr][pc] = 0
                    idx += 1
                else:
                    continue
                break
            else:
                yield ('fail', -1, -1, -1, copy.deepcopy(b))
                return
            continue

        # Try candidates with FC check
        placed = False
        for i, v in enumerate(vals):
            if is_valid(b, r, c, v):
                b[r][c] = v
                if forward_check_ok(b, r, c):
                    yield ('place', r, c, v, copy.deepcopy(b))
                    stack.append((r, c, vals, i))
                    placed = True
                    break
                b[r][c] = 0

        # If no placement was possible, backtrack
        if not placed:
            while stack:
                pr, pc, pvals, idx = stack.pop()
                b[pr][pc] = 0
                yield ('backtrack', pr, pc, -1, copy.deepcopy(b))

                idx += 1
                while idx < len(pvals):
                    v = pvals[idx]
                    if is_valid(b, pr, pc, v):
                        b[pr][pc] = v
                        if forward_check_ok(b, pr, pc):
                            yield ('place', pr, pc, v, copy.deepcopy(b))
                            stack.append((pr, pc, pvals, idx))
                            break
                        b[pr][pc] = 0
                    idx += 1
                else:
                    continue
                break
            else:
                yield ('fail', -1, -1, -1, copy.deepcopy(b))
                return


# ---------------- Visualization (matplotlib grid) -------------------------------------------
def plot_board(board, focus=None, title="Sudoku"):
    """Draw the Sudoku board with optional highlighted cell."""
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.zeros((9, 9)), cmap="Greys")

    # Grid lines
    ax.set_xticks(np.arange(-.5, 9, 1))
    ax.set_yticks(np.arange(-.5, 9, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="major", color="black", linewidth=1)

    # Bold 3×3 grid
    for k in (2.5, 5.5, 8.5):
        ax.axhline(k, color="black", linewidth=3)
        ax.axvline(k, color="black", linewidth=3)

    # Numbers
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v != 0:
                ax.text(c, r, str(v), ha="center", va="center", fontsize=16)

    # Highlight current cell
    if focus:
        fr, fc = focus
        ax.add_patch(plt.Rectangle((fc - 0.5, fr - 0.5), 1, 1, fill=False, linewidth=3, color="red"))

    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)


# ---------------- Streamlit UI State & Controls ---------------------------------------------
if "board" not in st.session_state:
    st.session_state.board = copy.deepcopy(puzzle_default)
if "steps" not in st.session_state:
    st.session_state.steps = list(solve_with_steps_fc(copy.deepcopy(st.session_state.board)))
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "auto" not in st.session_state:
    st.session_state.auto = False
if "delay" not in st.session_state:
    st.session_state.delay = 0.12

# Auto-play behavior
if st.session_state.auto and st.session_state.idx < len(st.session_state.steps) - 1:
    time.sleep(st.session_state.delay)
    st.session_state.idx += 1
    st.rerun()

# Buttons
c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])

if c1.button("Reset"):
    st.session_state.board = copy.deepcopy(puzzle_default)
    st.session_state.steps = list(solve_with_steps_fc(copy.deepcopy(st.session_state.board)))
    st.session_state.idx = 0
    st.session_state.auto = False

c2.button("Prev",
          disabled=st.session_state.idx <= 0,
          on_click=lambda: st.session_state.update(idx=st.session_state.idx - 1))

c3.button("Next",
          disabled=st.session_state.idx >= len(st.session_state.steps) - 1,
          on_click=lambda: st.session_state.update(idx=st.session_state.idx + 1))

c4.button("Run",
          disabled=st.session_state.idx >= len(st.session_state.steps) - 1,
          on_click=lambda: st.session_state.update(idx=st.session_state.idx + 1))

def _toggle_auto():
    st.session_state.auto = not st.session_state.auto
c5.button("Auto", on_click=_toggle_auto)

def _jump_end():
    st.session_state.idx = len(st.session_state.steps) - 1
    st.session_state.auto = False

c6.button("Solve to End", on_click=_jump_end)

st.session_state.delay = st.slider("Auto speed (sec/step)", 0.02, 0.5, st.session_state.delay, 0.01)

# Current frame
action, r, c, v, snapshot = st.session_state.steps[st.session_state.idx]
plot_board(snapshot,
           focus=(r, c) if r >= 0 else None,
           title=f"MRV + FC | {action} | Step {st.session_state.idx+1}/{len(st.session_state.steps)}")
