"""
Q @ Kᵀ — row-by-row attention score animation
Render:
    uv run manim -ql c_attention.py Attention
    uv run manim -qh c_attention.py Attention
"""

from manim import (
    DOWN,
    ITALIC,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0D0D0D"
TEXT_COL = "#DEDEDE"
STEP_COL = "#8c33b8"
Q_COL = "#4A9EFF"
K_COL = "#FF6B6B"
Q_ROW = ["#4A9EFF", "#50C878", "#FF6B6B", "#FFB347"]

# ── Model dims ────────────────────────────────────────────────────────────────
N, K = 4, 2
STEP_BUFF = 1.2

# Cell geometry
LCW = LCH = 0.44  # cell width / height
LCB = 0.04  # cell gap/buff

# Raw attention scores — used to drive cell opacity in A
RAW = [
    [0.90, 0.35, 0.15, 0.25],
    [0.20, 0.85, 0.30, 0.10],
    [0.12, 0.22, 0.88, 0.28],
    [0.30, 0.12, 0.25, 0.82],
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def mkgrid(nrows, ncols, cw, ch, cb, color, alpha=0.25, sw=0.8):
    """Flat VGroup of Rectangles in row-major order. Cell (i,j) → grid[i*ncols+j]."""
    cells = VGroup()
    for i in range(nrows):
        for j in range(ncols):
            c = Rectangle(
                width=cw,
                height=ch,
                color=color,
                fill_color=color,
                fill_opacity=alpha,
                stroke_width=sw,
            )
            c.move_to(RIGHT * j * (cw + cb) + DOWN * i * (ch + cb))
            cells.add(c)
    return cells


def mlbl(text, ref, fs=14, c=TEXT_COL, d=LEFT, b=0.18):
    lbl = Text(text, font_size=fs, color=c)
    lbl.next_to(ref, d, buff=b)
    return lbl


def slbl(text):
    return Text(text, font_size=21, color=STEP_COL, slant=ITALIC)


def grow(grid, row_i, ncols):
    """Cells in row i of a flat grid with ncols columns."""
    return [grid[row_i * ncols + j] for j in range(ncols)]


def gcol(grid, col_j, nrows, ncols):
    """Cells in column j of a flat grid."""
    return [grid[r * ncols + col_j] for r in range(nrows)]


def gc(grid, row_i, col_j, ncols):
    """Single cell at (i, j)."""
    return grid[row_i * ncols + col_j]


# ── Scene ─────────────────────────────────────────────────────────────────────


class Attention(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.p4_dot_product()

    # ── p4: compute A = Q Kᵀ (row by row) ───────────────────────────────────
    def p4_dot_product(self):
        # Ghost A, large Q and Kᵀ in standard matmul layout
        a = mkgrid(N, N, LCW, LCH, LCB, "#1c1c1c", alpha=0.12, sw=0.55)
        a.move_to(RIGHT * 0.9 + DOWN * 0.1)

        q = mkgrid(N, K, LCW, LCH, LCB, Q_COL)
        q.next_to(a, LEFT, buff=0.18)
        q.align_to(a, UP)

        kt = mkgrid(K, N, LCW, LCH, LCB, K_COL)
        kt.next_to(a, UP, buff=0.18)
        kt.align_to(a, LEFT)

        lq = mlbl("Q", q, fs=17, c=Q_COL, d=LEFT, b=0.22)
        lkt = mlbl("Kᵀ", kt, fs=17, c=K_COL, d=UP, b=0.14)

        nst = slbl(
            "fix one query -> dot with every key -> attention pattern for that query"
        ).to_edge(DOWN, buff=STEP_BUFF)

        self.play(FadeIn(q), FadeIn(lq), FadeIn(kt), FadeIn(lkt), run_time=0.5)
        self.play(FadeIn(a), FadeIn(nst, shift=UP * 0.08), run_time=0.4)
        self.wait(0.3)

        # ── Row-by-row dot product ────────────────────────────────────────────
        for i in range(N):
            qcol = Q_ROW[i]
            q_row = grow(q, i, K)

            # Highlight query row i
            self.play(
                *[c.animate.set_fill(color=qcol, opacity=0.78) for c in q_row],
                run_time=0.20,
            )

            for j in range(N):
                kt_c = gcol(kt, j, K, N)
                a_cell = gc(a, i, j, N)
                op = 0.13 + 0.67 * (RAW[i][j] / max(RAW[i]))

                # Light up Kᵀ column + reveal A[i,j] simultaneously
                self.play(
                    *[c.animate.set_fill(color=qcol, opacity=0.55) for c in kt_c],
                    a_cell.animate.set_fill(color=qcol, opacity=op),
                    run_time=0.20,
                )
                # Reset Kᵀ column
                self.play(
                    *[c.animate.set_fill(color=K_COL, opacity=0.25) for c in kt_c],
                    run_time=0.08,
                )

            # Settle query row to medium opacity
            self.play(
                *[c.animate.set_fill(color=qcol, opacity=0.42) for c in q_row],
                run_time=0.12,
            )

        self.wait(0.5)
        self._q = q
        self._kt = kt
        self._a = a
        self._lq = lq
        self._lkt = lkt
        self._st = nst
