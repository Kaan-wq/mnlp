"""
Post 1 — Tokenization
Render:
    uv run manim -ql a_tokenization.py Tokenization
    uv run manim -qh a_tokenization.py Tokenization
"""

from manim import (
    DOWN,
    ITALIC,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    FadeIn,
    FadeOut,
    LaggedStart,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    config,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0D0D0D"
ACCENT = "#4A9EFF"
TOKEN = "#50C878"
ID_COL = "#FFD700"
TEXT_COL = "#DEDEDE"
DIM_COL = "#2a2a2a"
DIM_TEXT = "#555555"
STEP_COL = "#8c33b8"

S1 = "Domain expansion : Infinite Void !"
S2 = "Domain expansion : Self-Embodiment of Perfection !"

STEP_BUFF = 1.2

# ── Helpers ───────────────────────────────────────────────────────────────────


def char_box(ch, color=ACCENT, font_size=16):
    lbl = Text(ch, color=WHITE, font_size=font_size, font="Monospace")
    box = RoundedRectangle(
        corner_radius=0.07,
        width=max(lbl.width + 0.18, 0.30),
        height=0.32,
        color=color,
        fill_color=color,
        fill_opacity=0.15,
        stroke_width=1.2,
    )
    lbl.move_to(box)
    return VGroup(box, lbl)


def token_box(text, color=TOKEN, font_size=17):
    lbl = Text(text, color=WHITE, font_size=font_size, font="Monospace")
    box = RoundedRectangle(
        corner_radius=0.09,
        width=max(lbl.width + 0.22, 0.34),
        height=0.36,
        color=color,
        fill_color=color,
        fill_opacity=0.20,
        stroke_width=1.4,
    )
    lbl.move_to(box)
    return VGroup(box, lbl)


def id_box(text, id_val, font_size=16):
    lbl = Text(text, color=WHITE, font_size=font_size, font="Monospace")
    id_t = Text(str(id_val), color=ID_COL, font_size=font_size - 2, font="Monospace")
    box = RoundedRectangle(
        corner_radius=0.09,
        width=max(lbl.width + 0.22, 0.34),
        height=0.36,
        color=TOKEN,
        fill_color=TOKEN,
        fill_opacity=0.20,
        stroke_width=1.4,
    )
    lbl.move_to(box)
    return VGroup(box, lbl), id_t


def step_label(text):
    return Text(text, font_size=21, color=STEP_COL, slant=ITALIC)


def make_char_row(sentence, color=ACCENT, font_size=16, buff=0.05):
    chars = [("·" if c == " " else c) for c in sentence]
    return VGroup(
        *[char_box(c, color=color, font_size=font_size) for c in chars]
    ).arrange(RIGHT, buff=buff)


# ── Tokenization data ─────────────────────────────────────────────────────────
# After BPE, these are the final tokens for each sentence
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

# Toy IDs (illustrative)
S1_IDS = [4230, 1, 323, 6839, 295, 1, 25, 1, 554, 4125, 578, 1, 13038, 312, 1, 0]
S2_IDS = [
    4230,
    1,
    323,
    6839,
    295,
    1,
    25,
    1,
    12376,
    12,
    2840,
    2127,
    748,
    434,
    1,
    286,
    1,
    2448,
    4828,
    295,
    1,
    0,
]


# ── Scene ─────────────────────────────────────────────────────────────────────


class Tokenization(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.phase_1_corpus()
        self.phase_2_char_split()
        self.phase_3_merge()
        self.phase_4_ids()

    # ── Phase 1: Show corpus ───────────────────────────────────────────────────

    def phase_1_corpus(self):
        s1 = Text(f'"{S1}"', font_size=22, color=TEXT_COL)
        s2 = Text(f'"{S2}"', font_size=22, color=TEXT_COL)
        VGroup(s1, s2).arrange(DOWN, buff=0.55).move_to(ORIGIN + UP * 0.2)

        self.play(
            LaggedStart(
                FadeIn(s1, shift=UP * 0.12),
                FadeIn(s2, shift=UP * 0.12),
                lag_ratio=0.35,
                run_time=1.0,
            )
        )
        self.wait(0.8)

        # Store for next phase
        self._s1_text = s1
        self._s2_text = s2

    # ── Phase 2: Character split ───────────────────────────────────────────────

    def phase_2_char_split(self):
        s1 = self._s1_text
        s2 = self._s2_text

        # Step label appears below
        step = step_label("character-level split").to_edge(DOWN, buff=STEP_BUFF)
        self.play(FadeIn(step, shift=UP * 0.08), run_time=0.4)
        self.wait(0.3)

        # Build char rows (initially invisible, positioned where sentences are)
        row1 = make_char_row(S1, color=ACCENT, font_size=14, buff=0.04)
        row2 = make_char_row(S2, color=ACCENT, font_size=14, buff=0.04)

        # Scale to fit screen width
        max_w = config.frame_width - 1.2
        for row in (row1, row2):
            if row.width > max_w:
                row.scale(max_w / row.width)

        # Position at same vertical spots as sentences
        row1.move_to(s1.get_center())
        row2.move_to(s2.get_center())

        # Cross-fade sentences → char rows
        self.play(
            FadeOut(s1),
            FadeOut(s2),
            run_time=0.3,
        )
        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(b, scale=0.7) for b in row1], lag_ratio=0.02),
                LaggedStart(*[FadeIn(b, scale=0.7) for b in row2], lag_ratio=0.02),
                lag_ratio=0.25,
                run_time=1.6,
            )
        )
        self.wait(0.7)

        self._row1 = row1
        self._row2 = row2
        self._step_label = step

    # ── Phase 3: Merge → tokens ────────────────────────────────────────────────

    def phase_3_merge(self):
        row1 = self._row1
        row2 = self._row2
        old_step = self._step_label

        # New step label
        new_step = step_label("learn and apply merge rules").to_edge(
            DOWN, buff=STEP_BUFF
        )
        self.play(
            FadeOut(old_step, shift=DOWN * 0.1),
            FadeIn(new_step, shift=UP * 0.1),
            run_time=0.4,
        )
        self.wait(0.3)

        # Build token rows
        tok_row1 = VGroup(*[token_box(t) for t in S1_TOKENS]).arrange(RIGHT, buff=0.07)
        tok_row2 = VGroup(*[token_box(t) for t in S2_TOKENS]).arrange(RIGHT, buff=0.07)

        max_w = config.frame_width - 1.2
        for row in (tok_row1, tok_row2):
            if row.width > max_w:
                row.scale(max_w / row.width)

        tok_row1.move_to(row1.get_center())
        tok_row2.move_to(row2.get_center())

        # Transform char rows → token rows
        self.play(
            LaggedStart(
                ReplacementTransform(row1, tok_row1),
                ReplacementTransform(row2, tok_row2),
                lag_ratio=0.2,
                run_time=1.0,
            )
        )
        self.wait(0.8)

        self._tok_row1 = tok_row1
        self._tok_row2 = tok_row2
        self._step_label = new_step

    # ── Phase 4: Token IDs ─────────────────────────────────────────────────────

    def phase_4_ids(self):
        tok_row1 = self._tok_row1
        tok_row2 = self._tok_row2
        old_step = self._step_label

        # New step label
        new_step = step_label("map each token to its integer ID").to_edge(
            DOWN, buff=STEP_BUFF
        )
        self.play(
            FadeOut(old_step, shift=DOWN * 0.1),
            FadeIn(new_step, shift=UP * 0.1),
            run_time=0.4,
        )
        self.wait(0.3)

        # Build ID rows — same boxes, add ID number below each
        def make_id_row(tok_row, tokens, ids):
            new_boxes = VGroup()
            id_labels = VGroup()
            for _, (tok, id_val) in enumerate(zip(tokens, ids, strict=True)):
                b = id_box(tok, id_val)
                new_boxes.add(b[0])
                id_labels.add(b[1])

            new_boxes.arrange(RIGHT, buff=0.07)
            max_w = config.frame_width - 1.2
            if new_boxes.width > max_w:
                new_boxes.scale(max_w / new_boxes.width)
            new_boxes.move_to(tok_row.get_center())

            # Position ID labels below each box
            for box, lbl in zip(new_boxes, id_labels, strict=True):
                lbl.next_to(box, DOWN, buff=0.12).scale(
                    new_boxes.height / (lbl.height * 3.5)
                )

            return new_boxes, id_labels

        id_boxes1, id_lbls1 = make_id_row(tok_row1, S1_TOKENS, S1_IDS)
        id_boxes2, id_lbls2 = make_id_row(tok_row2, S2_TOKENS, S2_IDS)

        # Swap token rows for ID-labeled rows
        self.play(
            LaggedStart(
                ReplacementTransform(tok_row1, id_boxes1),
                ReplacementTransform(tok_row2, id_boxes2),
                lag_ratio=0.2,
                run_time=0.8,
            )
        )

        # IDs rain in simultaneously
        all_ids = list(id_lbls1) + list(id_lbls2)
        self.play(
            LaggedStart(
                *[FadeIn(ll, shift=DOWN * 0.10) for ll in all_ids],
                lag_ratio=0.02,
                run_time=1.2,
            )
        )
        self.wait(0.8)

        # Closing note
        note = step_label("A sentence is now a sequence of integers.").to_edge(
            DOWN, buff=STEP_BUFF
        )
        self.play(
            FadeOut(new_step, shift=DOWN * 0.1),
            FadeIn(note, shift=UP * 0.1),
            run_time=0.4,
        )
        self.wait(1.2)
        self.play(
            FadeOut(VGroup(id_boxes1, id_boxes2, id_lbls1, id_lbls2, note)),
            run_time=0.5,
        )
