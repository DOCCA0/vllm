"""Polish the manually shortened defense deck without rebuilding it."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DECK = Path(__file__).with_name("eplb_migration_batching.pptx")


def remove_shape(shape) -> None:
    shape._element.getparent().remove(shape._element)


def remove_page_subtitles(prs: Presentation) -> int:
    """Remove the single-line explanatory text directly below page titles."""
    removed = 0
    for slide in prs.slides:
        for shape in reversed(list(slide.shapes)):
            if not shape.has_text_frame or not shape.text.strip():
                continue
            x = shape.left / Inches(1)
            y = shape.top / Inches(1)
            width = shape.width / Inches(1)
            if 0.45 <= x <= 0.7 and 0.95 <= y <= 1.12 and width >= 8:
                remove_shape(shape)
                removed += 1
    return removed


def set_shape_font_size(shape, size: float) -> None:
    for paragraph in shape.text_frame.paragraphs:
        paragraph.font.size = Pt(size)
        for run in paragraph.runs:
            run.font.size = Pt(size)


def set_text(
    shape,
    text: str,
    size: float,
    color: RGBColor,
    *,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.14)
    frame.margin_right = Inches(0.14)
    frame.margin_top = Inches(0.08)
    frame.margin_bottom = Inches(0.06)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_card(slide, x: float, title: str, body: str, accent: RGBColor) -> None:
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(2.75),
        Inches(3.45),
        Inches(2.45),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card.line.color.rgb = accent
    card.line.width = Pt(1.6)

    title_box = slide.shapes.add_textbox(
        Inches(x + 0.22), Inches(2.95), Inches(3.0), Inches(0.42)
    )
    set_text(title_box, title, 15, accent, bold=True)

    body_box = slide.shapes.add_textbox(
        Inches(x + 0.22), Inches(3.42), Inches(3.0), Inches(1.48)
    )
    set_text(body_box, body, 13.5, RGBColor(31, 48, 70))


def rebuild_profiling_method(prs: Presentation) -> None:
    """Explain how scheduler cost is compared with clean serving benefit."""
    for slide in prs.slides:
        if not any(
            shape.has_text_frame and shape.text.strip() == "Profiling Method"
            for shape in slide.shapes
        ):
            continue
        for shape in reversed(list(slide.shapes)[4:]):
            remove_shape(shape)

        navy = RGBColor(20, 52, 91)
        blue = RGBColor(47, 111, 174)
        purple = RGBColor(125, 72, 161)
        orange = RGBColor(230, 126, 34)
        green = RGBColor(19, 143, 95)

        question = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.7),
            Inches(1.68),
            Inches(12.0),
            Inches(0.72),
        )
        question.fill.solid()
        question.fill.fore_color.rgb = RGBColor(232, 241, 250)
        question.line.color.rgb = blue
        question.line.width = Pt(1.4)
        set_text(
            question,
            "PURPOSE   Is the added scheduler cost small compared with "
            "the serving time saved?",
            17,
            navy,
            bold=True,
            align=PP_ALIGN.CENTER,
        )

        add_card(
            slide,
            0.7,
            "1   PROFILE COST",
            "Add an NVTX range around the cycle-wide greedy scheduler.\n\n"
            "Nsight Systems records every call on all four Ray workers.",
            purple,
        )
        add_card(
            slide,
            4.94,
            "2   AGGREGATE",
            "Each call builds schedules for all 48 MoE layers.\n\n"
            "Sum call time per rank and use the largest rank total.",
            orange,
        )
        add_card(
            slide,
            9.18,
            "3   MEASURE BENEFIT",
            "Run a clean async E2E A/B with identical settings and no profiler.\n\n"
            "Saved time = batching OFF − batching ON.",
            green,
        )

        for x in (4.32, 8.56):
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.CHEVRON,
                Inches(x),
                Inches(3.62),
                Inches(0.42),
                Inches(0.55),
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = blue
            arrow.line.fill.background()

        range_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.95),
            Inches(5.6),
            Inches(11.5),
            Inches(0.76),
        )
        range_box.fill.solid()
        range_box.fill.fore_color.rgb = RGBColor(238, 247, 242)
        range_box.line.color.rgb = green
        range_box.line.width = Pt(1.3)
        set_text(
            range_box,
            'SEARCH IN TRACE   "eplb: schedule migration batches"   '
            "→   compare scheduler cost / serving time saved",
            15,
            navy,
            bold=True,
            align=PP_ALIGN.CENTER,
        )


def enlarge_small_text(prs: Presentation) -> None:
    """Raise small labels while respecting compact tables and diagrams."""
    for slide_number, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame or not shape.text.strip():
                continue

            y = shape.top / Inches(1)
            explicit_sizes = [
                run.font.size.pt
                for paragraph in shape.text_frame.paragraphs
                for run in paragraph.runs
                if run.font.size is not None
            ]

            # Small section labels above the page title.
            if y < 0.35:
                if explicit_sizes and min(explicit_sizes) < 10:
                    set_shape_font_size(shape, 10)
                continue

            # The compact navigation strip on the setup slide is decorative.
            if slide_number == 10 and y > 6.3:
                if explicit_sizes and min(explicit_sizes) < 9:
                    set_shape_font_size(shape, 9)
                continue

            minimum = {
                1: 11,
                3: 11.5,
                5: 11.5,
                6: 12,
                8: 12,
                9: 11.5,
                10: 11.5,
                11: 11.5,
                12: 11,
                13: 11,
                14: 11,
                15: 11,
                16: 12,
            }.get(slide_number, 11)

            if explicit_sizes and min(explicit_sizes) < minimum:
                set_shape_font_size(shape, minimum)


def main() -> None:
    prs = Presentation(DECK)
    removed = remove_page_subtitles(prs)
    rebuild_profiling_method(prs)
    enlarge_small_text(prs)
    prs.save(DECK)
    print(f"Removed {removed} page subtitles and enlarged small text.")


if __name__ == "__main__":
    main()
