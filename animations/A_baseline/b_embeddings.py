"""
Post 1 — Embeddings
Render:
    uv run manim -ql b_embeddings.py Embeddings
    uv run manim -qh b_embeddings.py Embeddings
"""

from manim import (
    DOWN,
    ITALIC,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    Brace,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    config,
)

# ── Palette (shared with a_tokenization.py) ───────────────────────────────────
BG = "#0D0D0D"
ACCENT = "#4A9EFF"
TOKEN = "#50C878"
ID_COL = "#FFD700"
TEXT_COL = "#DEDEDE"
DIM_COL = "#111111"
DIM_TEXT = "#555555"
STEP_COL = "#8c33b8"
VEC_COL = "#FF6B6B"

# ── Corpus data (same as a_tokenization.py) ───────────────────────────────────
S1_TOKENS = [
    "Domain",
    "·",
    "ex",
    "pan",
    "sion",
    "·",
    ":",
    "·",
    "In",
    "fin",
    "ite",
    "·",
    "Vo",
    "id",
    "·",
    "!",
]
S1_IDS = [4230, 1, 323, 6839, 295, 1, 25, 1, 554, 4125, 578, 1, 13038, 312, 1, 0]

S2_TOKENS = [
    "Domain",
    "·",
    "ex",
    "pan",
    "sion",
    "·",
    ":",
    "·",
    "Self",
    "-",
    "Em",
    "bo",
    "di",
    "ment",
    "·",
    "of",
    "·",
    "Per",
    "fec",
    "tion",
    "·",
    "!",
]

N = len(S1_TOKENS)  # 16
VOCAB_SIZE = 50_257
D_DIM = 512
STEP_BUFF = 0.7


# ── Helpers ───────────────────────────────────────────────────────────────────


def step_label(text):
    return Text(text, font_size=21, color=STEP_COL, slant=ITALIC)


def token_box_h(text, color=TOKEN, font_size=14):
    """Horizontal token box — used in Phase 1 overview."""
    lbl = Text(text, color=WHITE, font_size=font_size, font="Monospace")
    box = RoundedRectangle(
        corner_radius=0.07,
        width=max(lbl.width + 0.18, 0.30),
        height=0.30,
        color=color,
        fill_color=color,
        fill_opacity=0.18,
        stroke_width=1.2,
    )
    lbl.move_to(box)
    return VGroup(box, lbl)


def id_box_v(id_val, font_size=13):
    """Compact vertical box showing an integer ID."""
    lbl = Text(str(id_val), color=ID_COL, font_size=font_size, font="Monospace")
    box = RoundedRectangle(
        corner_radius=0.06,
        width=0.72,
        height=0.27,
        color=TOKEN,
        fill_color=TOKEN,
        fill_opacity=0.15,
        stroke_width=1.0,
    )
    lbl.move_to(box)
    return VGroup(box, lbl)


def make_bracket(height, side="left", color=TEXT_COL, stroke_w=1.5, serif=0.12):
    """Square bracket [ or ] built from Line objects."""
    direction = RIGHT if side == "left" else LEFT
    vert = Line(UP * height / 2, DOWN * height / 2, color=color, stroke_width=stroke_w)
    top_serif = Line(
        vert.get_start(),
        vert.get_start() + direction * serif,
        color=color,
        stroke_width=stroke_w,
    )
    bot_serif = Line(
        vert.get_end(),
        vert.get_end() + direction * serif,
        color=color,
        stroke_width=stroke_w,
    )
    return VGroup(vert, top_serif, bot_serif)


# ── Scene ─────────────────────────────────────────────────────────────────────


class Embeddings(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.phase_1_corpus()
        self.phase_2_column()
        self.phase_3_table()
        self.phase_4_output()
        self.phase_5_closing()

    # ── Phase 1: Show both tokenized sequences ─────────────────────────────────
    def phase_1_corpus(self):
        max_w = config.frame_width - 1.4
        row1 = VGroup(*[token_box_h(t) for t in S1_TOKENS]).arrange(RIGHT, buff=0.05)
        row2 = VGroup(*[token_box_h(t) for t in S2_TOKENS]).arrange(RIGHT, buff=0.05)
        for row in (row1, row2):
            if row.width > max_w:
                row.scale(max_w / row.width)
        VGroup(row1, row2).arrange(DOWN, buff=0.55).move_to(ORIGIN + UP * 0.15)

        step = step_label("previously tokenized sequences").to_edge(
            DOWN, buff=STEP_BUFF
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(b, scale=0.8) for b in row1], lag_ratio=0.03),
                LaggedStart(*[FadeIn(b, scale=0.8) for b in row2], lag_ratio=0.03),
                lag_ratio=0.4,
                run_time=1.4,
            )
        )
        self.play(FadeIn(step, shift=UP * 0.08), run_time=0.4)
        self.wait(0.7)

        self._row1 = row1
        self._row2 = row2
        self._step = step

    # ── Phase 2: Focus on S1, rearrange as column of IDs ──────────────────────
    def phase_2_column(self):
        row1, row2, old_step = self._row1, self._row2, self._step

        self.play(
            FadeOut(old_step, shift=DOWN * 0.1),
            FadeOut(row2),
            run_time=0.5,
        )
        self.wait(0.2)

        # Build vertical ID column
        col = VGroup(*[id_box_v(i) for i in S1_IDS]).arrange(DOWN, buff=0.05)
        col.scale(4.0 / col.height)
        col.move_to(LEFT * 3.8)

        # Transform horizontal token row → vertical ID column
        self.play(ReplacementTransform(row1, col), run_time=0.9)

        # Square brackets + N label
        lb = make_bracket(col.height * 1.06, side="left")
        rb = make_bracket(col.height * 1.06, side="right")
        lb.next_to(col, LEFT, buff=0.10)
        rb.next_to(col, RIGHT, buff=0.10)
        n_lbl = Text(f"N = {N}", font_size=15, color=TEXT_COL)
        n_lbl.next_to(lb, LEFT, buff=0.12)

        d_brace_col = Brace(col, UP, color=TEXT_COL, buff=0.08)
        d_lbl_col = Text("D = 1", font_size=14, color=TEXT_COL)
        d_lbl_col.next_to(d_brace_col, UP, buff=0.06)

        self.play(
            FadeIn(lb),
            FadeIn(rb),
            FadeIn(n_lbl),
            FadeIn(d_brace_col),
            FadeIn(d_lbl_col),
            run_time=0.4,
        )
        self.wait(0.6)

        self._col = col
        self._col_deco = VGroup(lb, rb, n_lbl, d_brace_col, d_lbl_col)
        self._step = None

    # ── Phase 3: Embedding table (black box) ───────────────────────────────────
    def phase_3_table(self):
        col, old_step = self._col, self._step

        new_step = step_label(
            "each integer ID indexes a learnable row in the embedding table"
        ).to_edge(DOWN, buff=STEP_BUFF)
        plays = [FadeIn(new_step, shift=UP * 0.1)]
        if old_step is not None:
            plays.append(FadeOut(old_step, shift=DOWN * 0.1))
        self.play(*plays, run_time=0.4)
        self.wait(0.2)

        # Dark block
        tbl_h = 4.0
        tbl_w = 2.8
        table = RoundedRectangle(
            corner_radius=0.14,
            width=tbl_w,
            height=tbl_h,
            color="#333333",
            fill_color=DIM_COL,
            fill_opacity=1.0,
            stroke_width=1.5,
        )
        table.move_to(ORIGIN)

        # Rows inside suggesting a lookup table
        n_grid = 6
        g_h = tbl_h / (n_grid + 2) * 0.60
        grid = VGroup(
            *[
                Rectangle(
                    width=tbl_w * 0.70,
                    height=g_h,
                    color=TOKEN,
                    fill_color=TOKEN,
                    fill_opacity=0.09 + i * 0.025,
                    stroke_width=0.7,
                )
                for i in range(n_grid)
            ]
        ).arrange(DOWN, buff=g_h * 0.28)
        grid.move_to(table.get_center() + UP * tbl_h * 0.05)
        dots = Text("⋮", font_size=18, color=DIM_TEXT).next_to(grid, DOWN, buff=0.07)

        title = Text("Embedding Table", font_size=17, color=WHITE)
        title.next_to(table, UP, buff=0.16)

        dim_lbl = Text(f"({VOCAB_SIZE:,}  ×  {D_DIM})", font_size=14, color=TEXT_COL)
        dim_lbl.next_to(table, DOWN, buff=0.14)

        note_lbl = Text(
            "one learnable row per token ID", font_size=12, color=DIM_TEXT, slant=ITALIC
        )
        note_lbl.next_to(dim_lbl, DOWN, buff=0.07)

        arrow_in = Arrow(
            col.get_right() + RIGHT * 0.05,
            table.get_left() - RIGHT * 0.05,
            color=ACCENT,
            stroke_width=2.0,
            max_tip_length_to_length_ratio=0.10,
        )

        self.play(FadeIn(table), run_time=0.35)
        self.play(
            LaggedStart(
                *[FadeIn(r, shift=RIGHT * 0.04) for r in grid],
                lag_ratio=0.09,
                run_time=0.55,
            )
        )
        self.play(FadeIn(dots), run_time=0.2)
        self.play(
            FadeIn(title, shift=DOWN * 0.06),
            FadeIn(dim_lbl, shift=UP * 0.06),
            FadeIn(note_lbl, shift=UP * 0.06),
            run_time=0.5,
        )
        self.play(FadeIn(arrow_in), run_time=0.4)
        self.wait(0.6)

        self._table = table
        self._table_inner = VGroup(grid, dots)
        self._table_labels = VGroup(title, dim_lbl, note_lbl)
        self._arrow_in = arrow_in
        self._step = new_step

    # ── Phase 4: Output (N × D) matrix ────────────────────────────────────────
    def phase_4_output(self):
        col, table, old_step = self._col, self._table, self._step

        if old_step is not None:
            self.play(FadeOut(old_step, shift=DOWN * 0.1), run_time=0.4)
        self.wait(0.2)

        bar_w = 2.4
        gap = table.get_left()[0] - col.get_right()[0]
        bar_h = 4.0 / N * 0.78
        bars = VGroup(
            *[
                Rectangle(
                    width=bar_w,
                    height=bar_h,
                    color=VEC_COL,
                    fill_color=VEC_COL,
                    fill_opacity=0.20 + (i % 4) * 0.06,
                    stroke_width=0.8,
                )
                for i in range(N)
            ]
        ).arrange(DOWN, buff=0.04)
        bars.move_to(
            [
                table.get_right()[0] + gap + bar_w / 2,
                0,
                0,
            ]
        )

        # Brackets
        lb_out = make_bracket(bars.height * 1.06, side="left")
        rb_out = make_bracket(bars.height * 1.06, side="right")
        lb_out.next_to(bars, LEFT, buff=0.10)
        rb_out.next_to(bars, RIGHT, buff=0.10)

        # D brace on top
        d_brace = Brace(bars, UP, color=TEXT_COL, buff=0.08)
        d_lbl = Text(f"D = {D_DIM}", font_size=14, color=TEXT_COL)
        d_lbl.next_to(d_brace, UP, buff=0.06)

        arrow_out = Arrow(
            table.get_right() + RIGHT * 0.05,
            bars.get_left() - RIGHT * 0.05,
            color=ACCENT,
            stroke_width=2.0,
            max_tip_length_to_length_ratio=0.10,
        )

        self.play(FadeIn(arrow_out), run_time=0.4)
        self.play(
            LaggedStart(
                *[FadeIn(b, shift=RIGHT * 0.10) for b in bars],
                lag_ratio=0.04,
                run_time=1.0,
            )
        )
        self.play(FadeIn(lb_out), FadeIn(rb_out), run_time=0.3)
        self.play(
            FadeIn(d_brace),
            FadeIn(d_lbl),
            run_time=0.5,
        )
        self.wait(1.0)

        self._bars = bars
        self._out_deco = VGroup(lb_out, rb_out, d_brace, d_lbl)
        self._arrow_out = arrow_out
        self._step = None

    # ── Phase 5: Closing ───────────────────────────────────────────────────────
    def phase_5_closing(self):
        self.wait(0.5)
        self.play(
            FadeOut(
                VGroup(
                    self._col,
                    self._col_deco,
                    self._table,
                    self._table_inner,
                    self._table_labels,
                    self._arrow_in,
                    self._bars,
                    self._out_deco,
                    self._arrow_out,
                )
            ),
            run_time=0.6,
        )
