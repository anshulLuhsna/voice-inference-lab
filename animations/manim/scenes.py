"""Article visuals tagged for Manim.

Numbers come from the paired overlap trace and the Moshi frame clock
in voice-inference-engineering-101.md.
"""

from manim import *

INK = "#F4EFE6"
MUTED = "#C9BBA6"
GOLD = "#C6A56A"
GOLD_SOFT = "#E4D3A8"
SAGE = "#A8B5A0"
CREAM = "#E7DCC8"
LINE = "#4A433A"
SHADE = "#C6A56A"

FONT = "EB Garamond"
UI = "Inter"


def label(text, size=28, color=INK, weight=NORMAL, font=UI):
    return Text(text, font=font, font_size=size, color=color, weight=weight)


class OverlapTimeline(Scene):
    """Sequential vs overlapped scheduling on one scale.

    Median first-audio times: 830.6 ms sequential, 478.6 ms overlapped.
    The shaded band is the 435 ms where synthesis and generation overlap
    (sentence-ready at 342 ms through generation end at 777 ms).
    """

    SCALE_MS = 900
    WIDTH = 12.8
    LEFT = -6.4

    def x_of(self, ms):
        return self.LEFT + (ms / self.SCALE_MS) * self.WIDTH

    def bar(self, t0, t1, y, color, height=0.62):
        width = max((t1 - t0) / self.SCALE_MS * self.WIDTH, 0.04)
        rect = Rectangle(
            width=width,
            height=height,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        rect.move_to([self.x_of(t0) + width / 2, y, 0])
        return rect

    def marker(self, ms, y, text, dy=0.42):
        x = self.x_of(ms)
        tick = Line([x, y - 0.28, 0], [x, y + dy, 0], color=GOLD_SOFT, stroke_width=2.5)
        name = label(text, 32, GOLD_SOFT)
        name.next_to(tick, UP, buff=0.08)
        return VGroup(tick, name)

    def construct(self):
        title = label("the overlap", 64, font=FONT)
        title.to_edge(UP, buff=0.22)
        sub = label("same work. the first sentence starts speaking while the model is still writing.", 32, MUTED)
        sub.scale_to_fit_width(12.4)
        sub.next_to(title, DOWN, buff=0.16)

        seq_y = 0.45
        ov_think_y = -1.05
        ov_speak_y = -2.25

        axis = VGroup()
        for ms in (0, 300, 600, 900):
            x = self.x_of(ms)
            tick = Line([x, -2.55, 0], [x, -2.32, 0], color=LINE, stroke_width=2)
            caption = label("900 ms" if ms == 900 else str(ms), 30, MUTED)
            caption.next_to(tick, DOWN, buff=0.08)
            axis.add(tick, caption)
        baseline = Line(
            [self.LEFT, -2.42, 0],
            [self.x_of(900), -2.42, 0],
            color=LINE,
            stroke_width=2,
        )

        seq_tag = label("sequential", 32, MUTED)
        seq_tag.move_to([-6.6, seq_y + 0.62, 0], aligned_edge=LEFT)
        ov_tag = label("overlapped", 32, MUTED)
        ov_tag.move_to([-6.6, ov_think_y + 0.62, 0], aligned_edge=LEFT)

        listen = self.bar(0, 188, seq_y, SAGE)
        think = self.bar(188, 685, seq_y, GOLD)
        speak = self.bar(685, 831, seq_y, CREAM)
        listen_l = label("listen", 30, "#1A1714")
        think_l = label("think", 30, "#1A1714")
        speak_l = label("speak", 30, "#1A1714")
        listen_l.move_to(listen)
        think_l.move_to(think)
        speak_l.move_to(speak)
        seq_audio = self.marker(830.6, seq_y, "830 ms")

        o_listen = self.bar(0, 187, ov_think_y, SAGE)
        o_think = self.bar(187, 777, ov_think_y, GOLD)
        o_speak = self.bar(342, 845, ov_speak_y, CREAM)
        o_listen_l = label("listen", 30, "#1A1714")
        o_think_l = label("think", 30, "#1A1714")
        o_speak_l = label("speak", 28, "#1A1714")
        o_listen_l.move_to(o_listen)
        o_think_l.move_to([self.x_of((187 + 342) / 2), ov_think_y, 0])
        o_speak_l.move_to([self.x_of((777 + 845) / 2), ov_speak_y, 0])

        shade = self.bar(342, 777, (ov_think_y + ov_speak_y) / 2, SHADE, height=1.9)
        shade.set_fill(SHADE, opacity=0.22)
        shade.set_z_index(0)
        for piece in (o_listen, o_think, o_speak, o_listen_l, o_think_l, o_speak_l):
            piece.set_z_index(2)
        shade.set_fill(SHADE, opacity=0.16)
        shade_l = label("435 ms overlap", 32, GOLD_SOFT)
        shade_l.move_to([self.x_of((342 + 777) / 2), ov_think_y + 0.7, 0])
        audio_x = self.x_of(478.6)
        audio_tick = Line(
            [audio_x, ov_speak_y - 0.31, 0],
            [audio_x, ov_speak_y + 0.31, 0],
            color="#1A1714",
            stroke_width=3,
        )
        audio_name = label("478 ms", 32, GOLD_SOFT)
        audio_name.next_to(o_speak, UP, buff=0.1)
        audio_name.align_to(o_speak, RIGHT)
        ov_audio = VGroup(audio_tick, audio_name)

        point = label("first audio 42% earlier. the turn still ends together.", 32, INK)
        point.scale_to_fit_width(12.2)
        point.to_edge(DOWN, buff=0.18)

        self.play(FadeIn(title, shift=UP * 0.15), FadeIn(sub), run_time=0.8)
        self.play(Create(baseline), FadeIn(axis), FadeIn(seq_tag), FadeIn(ov_tag), run_time=0.7)
        self.play(GrowFromEdge(listen, LEFT), FadeIn(listen_l), run_time=0.55)
        self.play(GrowFromEdge(think, LEFT), FadeIn(think_l), run_time=0.9)
        self.play(GrowFromEdge(speak, LEFT), FadeIn(speak_l), FadeIn(seq_audio), run_time=0.7)
        self.wait(0.35)
        self.play(GrowFromEdge(o_listen, LEFT), FadeIn(o_listen_l), run_time=0.45)
        self.play(GrowFromEdge(o_think, LEFT), FadeIn(o_think_l), run_time=1.0)
        self.play(
            GrowFromEdge(shade, LEFT),
            GrowFromEdge(o_speak, LEFT),
            FadeIn(o_speak_l),
            FadeIn(shade_l),
            run_time=1.05,
        )
        self.play(FadeIn(ov_audio), FadeIn(point, shift=UP * 0.1), run_time=0.7)
        self.wait(1.8)


class TwoStreamsOneClock(Scene):
    """Full duplex: 80 ms frames, output one frame behind, compute inside 49 ms."""

    def construct(self):
        title = label("two streams, one clock", 60, font=FONT)
        title.to_edge(UP, buff=0.22)
        sub = label("no request. no response. the output lags the input by one frame.", 32, MUTED)
        sub.scale_to_fit_width(12.6)
        sub.next_to(title, DOWN, buff=0.14)

        n = 7
        frame_w = 1.28
        gap = 0.1
        total = n * frame_w + (n - 1) * gap
        origin = -4.15 + frame_w / 2
        in_y = 0.7
        out_y = -1.15

        in_tag = label("audio in", 30, MUTED)
        out_tag = label("audio out", 30, MUTED)
        in_tag.move_to([-6.85, in_y, 0], aligned_edge=LEFT)
        out_tag.move_to([-6.85, out_y, 0], aligned_edge=LEFT)

        inputs = VGroup()
        outputs = VGroup()
        for i in range(n):
            x = origin + i * (frame_w + gap)
            box = Rectangle(
                width=frame_w,
                height=1.05,
                stroke_color=LINE,
                stroke_width=2,
                fill_color="#24201B",
                fill_opacity=1,
            )
            box.move_to([x, in_y, 0])
            num = label(str(i + 1), 36, INK)
            num.move_to(box)
            inputs.add(VGroup(box, num))

            out = Rectangle(
                width=frame_w,
                height=1.05,
                stroke_width=0,
                fill_color=GOLD,
                fill_opacity=1,
            )
            # one frame behind: output slot i sits under input slot i+1
            out.move_to([origin + (i + 1) * (frame_w + gap), out_y, 0])
            onum = label(str(i + 1), 36, "#1A1714")
            onum.move_to(out)
            outputs.add(VGroup(out, onum))

        # The last input has no output yet; drop the output that would fall off the right.
        outputs = outputs[: n - 1]

        sample = inputs[6][0]
        compute_w = frame_w * (49 / 80)
        compute = Rectangle(
            width=compute_w,
            height=0.28,
            stroke_width=0,
            fill_color=GOLD,
            fill_opacity=1,
        )
        compute.align_to(sample, LEFT)
        compute.align_to(sample, DOWN)
        compute.shift(UP * 0.08 + RIGHT * 0.06)
        compute_l = label("49 ms of compute inside an 80 ms frame", 34, GOLD_SOFT)
        compute_l.to_edge(DOWN, buff=1.15)

        play = Line(
            [inputs[0][0].get_left()[0] - 0.05, in_y + 0.85, 0],
            [inputs[0][0].get_left()[0] - 0.05, out_y - 0.85, 0],
            color=GOLD_SOFT,
            stroke_width=3,
        )

        note = label("the model finishes the frame before the world makes the next one.", 32, INK)
        note.scale_to_fit_width(12.6)
        note.to_edge(DOWN, buff=0.22)

        self.play(FadeIn(title, shift=UP * 0.12), FadeIn(sub), run_time=0.7)
        self.play(FadeIn(in_tag), FadeIn(out_tag), run_time=0.35)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.08) for g in inputs], lag_ratio=0.12), run_time=1.3)
        self.play(FadeIn(play), run_time=0.3)

        for i, group in enumerate(outputs):
            target_x = inputs[i + 1][0].get_center()[0]
            self.play(
                play.animate.set_x(target_x),
                FadeIn(group, shift=RIGHT * 0.05),
                run_time=0.38,
            )

        self.play(GrowFromEdge(compute, LEFT), FadeIn(compute_l), run_time=0.7)
        self.play(FadeIn(note, shift=UP * 0.08), run_time=0.5)
        self.wait(1.6)
