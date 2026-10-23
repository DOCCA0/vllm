"""Polish the manually shortened defense deck without rebuilding it."""

from __future__ import annotations

import copy
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DECK = Path(__file__).with_name("eplb_migration_batching.pptx")


def load_equations(formulas: dict[str, str]) -> dict[str, etree._Element]:
    """Convert LaTeX to editable Office Math through Pandoc."""
    markdown: list[str] = []
    for key, formula in formulas.items():
        markdown.extend([f"# {key}", "", f"$${formula}$$", "", "---", ""])

    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "equations.md"
        deck = Path(temp_dir) / "equations.pptx"
        source.write_text("\n".join(markdown), encoding="utf-8")
        subprocess.run(["pandoc", str(source), "-o", str(deck)], check=True)
        equations: dict[str, etree._Element] = {}
        namespace = {"a14": "http://schemas.microsoft.com/office/drawing/2010/main"}
        with zipfile.ZipFile(deck) as archive:
            for index, key in enumerate(formulas, start=1):
                root = etree.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
                math = root.find(".//a14:m", namespaces=namespace)
                if math is None:
                    raise RuntimeError(f"Pandoc did not generate Office Math: {key}")
                equations[key] = copy.deepcopy(math)
    return equations


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


def add_equation(
    slide,
    math: etree._Element,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: int,
    color: RGBColor,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    tx_body = box._element.txBody
    drawing_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    for paragraph in list(tx_body.findall(f"{{{drawing_ns}}}p")):
        tx_body.remove(paragraph)

    paragraph = etree.Element(f"{{{drawing_ns}}}p")
    paragraph_properties = etree.SubElement(paragraph, f"{{{drawing_ns}}}pPr")
    paragraph_properties.set("algn", "ctr")
    equation = copy.deepcopy(math)
    for run in equation.findall(f".//{{{math_ns}}}r"):
        properties = etree.Element(f"{{{drawing_ns}}}rPr")
        properties.set("lang", "en-US")
        properties.set("sz", str(size * 100))
        properties.set("dirty", "0")
        fill = etree.SubElement(properties, f"{{{drawing_ns}}}solidFill")
        rgb = etree.SubElement(fill, f"{{{drawing_ns}}}srgbClr")
        rgb.set("val", "".join(f"{channel:02X}" for channel in color))
        latin = etree.SubElement(properties, f"{{{drawing_ns}}}latin")
        latin.set("typeface", "Cambria Math")
        text = run.find(f"{{{math_ns}}}t")
        run.insert(run.index(text) if text is not None else len(run), properties)
    paragraph.append(equation)
    tx_body.append(paragraph)
    body_properties = tx_body.find(f"{{{drawing_ns}}}bodyPr")
    body_properties.set("anchor", "ctr")
    body_properties.set("wrap", "none")


def add_panel(slide, x: float, y: float, width: float, height: float, color) -> None:
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(255, 255, 255)
    panel.line.color.rgb = color
    panel.line.width = Pt(1.4)


def rebuild_ilp_formulation(prs: Presentation) -> None:
    """Present the README model with consistent symbols and editable math."""
    formulas = load_equations(
        {
            "objective": (
                r"\min T_{\mathrm{infer}},\qquad T_{\mathrm{infer}}\ge "
                r"T_0\left(1+\frac{L_{e,b}}{C_e(1-\rho)}\right),"
                r"\quad \forall e,b"
            ),
            "assign": r"\sum_{b\in B}x_{p,b}=1,\qquad \forall p\in P",
            "endpoint": (
                r"\sum_{p:r\in p}x_{p,b}\le 1,"
                r"\qquad \forall r,\;b"
            ),
            "load": r"L_{e,b}=\sum_{p\in P}w_p f_{p,e}x_{p,b}",
        }
    )
    for slide in prs.slides:
        if not any(
            shape.has_text_frame and shape.text.strip() == "ILP Formulation"
            for shape in slide.shapes
        ):
            continue
        for shape in reversed(list(slide.shapes)[3:]):
            remove_shape(shape)

        navy = RGBColor(20, 52, 91)
        ink = RGBColor(31, 48, 70)
        blue = RGBColor(47, 111, 174)
        purple = RGBColor(125, 72, 161)
        orange = RGBColor(230, 126, 34)
        green = RGBColor(19, 143, 95)

        add_panel(slide, 0.62, 1.68, 3.18, 4.87, blue)
        heading = slide.shapes.add_textbox(
            Inches(0.88), Inches(1.88), Inches(2.65), Inches(0.3)
        )
        set_text(heading, "VARIABLES & INPUTS", 12, blue, bold=True)

        terms = [
            ("wₚ", "experts in flow p"),
            ("xₚ,ᵦ", "flow p assigned to batch b"),
            ("yᵦ", "batch b is active"),
        ]
        for index, (symbol, meaning) in enumerate(terms):
            y = 2.38 + index * 0.84
            symbol_box = slide.shapes.add_textbox(
                Inches(0.9), Inches(y), Inches(0.82), Inches(0.42)
            )
            set_text(symbol_box, symbol, 19, navy, bold=True)
            meaning_box = slide.shapes.add_textbox(
                Inches(1.72), Inches(y - 0.01), Inches(1.78), Inches(0.48)
            )
            set_text(meaning_box, meaning, 12.5, ink)

        tie_break = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.88),
            Inches(5.15),
            Inches(2.66),
            Inches(1.0),
        )
        tie_break.fill.solid()
        tie_break.fill.fore_color.rgb = RGBColor(255, 244, 225)
        tie_break.line.color.rgb = orange
        set_text(
            tie_break,
            "TIE-BREAK ONLY\nAfter fixing the optimum, minimize active batches",
            11.5,
            ink,
            bold=True,
            align=PP_ALIGN.CENTER,
        )

        add_panel(slide, 4.08, 1.62, 8.5, 1.42, green)
        objective_label = slide.shapes.add_textbox(
            Inches(4.3), Inches(1.73), Inches(2.15), Inches(0.3)
        )
        set_text(objective_label, "PRIMARY OBJECTIVE", 11.5, green, bold=True)
        add_equation(
            slide,
            formulas["objective"],
            5.55,
            1.73,
            6.72,
            0.65,
            size=14,
            color=ink,
        )
        definitions = slide.shapes.add_textbox(
            Inches(4.31), Inches(2.5), Inches(7.95), Inches(0.32)
        )
        set_text(
            definitions,
            "T₀: baseline inference time   ·   Cₑ: link capacity   ·   "
            "Lₑ,ᵦ: migration load   ·   ρ: inference traffic ratio",
            10.5,
            ink,
            align=PP_ALIGN.CENTER,
        )

        constraint_rows = [
            (
                3.24,
                "1   Every flow runs exactly once",
                "assign",
                blue,
                15,
            ),
            (
                4.28,
                "2   At most one peer per rank and batch",
                "endpoint",
                purple,
                14,
            ),
            (
                5.32,
                "3   Migration load on each link",
                "load",
                orange,
                15,
            ),
        ]
        for y, label, key, color, size in constraint_rows:
            add_panel(slide, 4.08, y, 8.5, 0.92, color)
            label_box = slide.shapes.add_textbox(
                Inches(4.3), Inches(y + 0.1), Inches(2.85), Inches(0.62)
            )
            set_text(label_box, label, 11.5, color, bold=True)
            add_equation(
                slide,
                formulas[key],
                6.92,
                y + 0.13,
                5.3,
                0.58,
                size=size,
                color=ink,
            )


def remove_legacy_ilp_math(path: Path) -> None:
    """Remove superseded compatibility-layer equations from slide 6."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / path.name
        with (
            zipfile.ZipFile(path, "r") as source,
            zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target,
        ):
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "ppt/slides/slide6.xml":
                    root = etree.fromstring(data)
                    namespaces = {
                        "m": (
                            "http://schemas.openxmlformats.org/officeDocument/2006/math"
                        ),
                        "mc": (
                            "http://schemas.openxmlformats.org/markup-"
                            "compatibility/2006"
                        ),
                    }
                    for alternate in root.xpath(
                        ".//mc:AlternateContent[.//m:oMath]",
                        namespaces=namespaces,
                    ):
                        alternate.getparent().remove(alternate)
                    substitutions = {
                        "\u200b": "",
                        "\u2001": " ",
                        "\u2004": " ",
                        "\u2005": " ",
                        "\u2009": " ",
                    }
                    for text_node in root.xpath(".//m:t", namespaces=namespaces):
                        if text_node.text is None:
                            continue
                        for old, new in substitutions.items():
                            text_node.text = text_node.text.replace(old, new)
                    data = etree.tostring(
                        root,
                        xml_declaration=True,
                        encoding="UTF-8",
                        standalone=True,
                    )
                target.writestr(item, data)
        path.write_bytes(output.read_bytes())


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
    rebuild_ilp_formulation(prs)
    rebuild_profiling_method(prs)
    enlarge_small_text(prs)
    prs.save(DECK)
    remove_legacy_ilp_math(DECK)
    print(f"Removed {removed} page subtitles and enlarged small text.")


if __name__ == "__main__":
    main()
