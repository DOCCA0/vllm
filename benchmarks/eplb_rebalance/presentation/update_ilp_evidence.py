"""Update only the ILP evidence in the manually edited defense deck."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ILP = HERE.parent / "ILP"
DECK = HERE / "wucao_eplb.pptx"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
}


def textbox(text: str, x: float, y: float, w: float, h: float, size: int):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.text_frame.word_wrap = True
    shape.text_frame.margin_left = 0
    shape.text_frame.margin_right = 0
    shape.text_frame.margin_top = 0
    shape.text_frame.margin_bottom = 0
    shape.text = text
    for paragraph in shape.text_frame.paragraphs:
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = RGBColor(28, 39, 53)
    return copy.deepcopy(shape._element)


def append_shape(root, shape) -> None:
    tree = root.find("p:cSld/p:spTree", NS)
    ids = [int(item.get("id")) for item in root.findall(".//p:cNvPr", NS)]
    properties = shape.find("p:nvSpPr/p:cNvPr", NS)
    properties.set("id", str(max(ids) + 1))
    properties.set("name", f"ILP evidence {max(ids) + 1}")
    tree.append(shape)


def main() -> None:
    problem = json.loads((ILP / "example_hotspot.json").read_text())
    result = json.loads((ILP / "example_hotspot_solution.json").read_text())
    off_time = f"{result['unbatched_lower_bound']['inference_time']:.2f}"
    on_time = f"{result['optimal']['inference_time']:.2f}"
    with zipfile.ZipFile(DECK) as archive:
        entries = archive.infolist()
        data = {entry.filename: archive.read(entry) for entry in entries}
    roots = {
        page: etree.fromstring(data[f"ppt/slides/slide{page}.xml"]) for page in (7, 8)
    }
    for root in roots.values():
        for shape in list(root.findall("p:cSld/p:spTree/p:sp", NS)):
            name = shape.find("p:nvSpPr/p:cNvPr", NS).get("name", "")
            text = "".join(shape.xpath(".//a:t/text()", namespaces=NS))
            offset = shape.find("p:spPr/a:xfrm/a:off", NS)
            legacy_equation = (
                name == "TextBox 1"
                and offset is not None
                and offset.get("y") == str(Inches(6.65))
            )
            if (
                name.startswith("ILP evidence ")
                or text.startswith("Bandwidths: DEMI,")
                or text.startswith("Capacities: DEMI (")
                or legacy_equation
            ):
                shape.getparent().remove(shape)
    # Keep the rest of the ZIP entries unchanged, including equations, comments,
    # animations and all of the user's other slides.
    image_sources = {20: "links", 7: "before", 9: "batch0", 10: "batch1", 12: "batch2"}
    slide = Presentation(DECK).slides[6]
    for shape in slide.shapes:
        if shape.shape_id not in image_sources:
            continue
        source = ILP / f"example_hotspot_{image_sources[shape.shape_id]}.png"
        part = slide.part.related_part(shape._element.blipFill.blip.rEmbed)
        data[str(part.partname).lstrip("/")] = source.read_bytes()
        width, height = Image.open(source).size
        scale = min(shape.width / width, shape.height / height)
        new_width, new_height = round(width * scale), round(height * scale)
        node = roots[7].xpath(
            f'.//p:pic[p:nvPicPr/p:cNvPr[@id="{shape.shape_id}"]]',
            namespaces=NS,
        )[0]
        transform = node.find("p:spPr/a:xfrm", NS)
        transform.find("a:off", NS).set(
            "x", str(shape.left + (shape.width - new_width) // 2)
        )
        transform.find("a:off", NS).set(
            "y", str(shape.top + (shape.height - new_height) // 2)
        )
        transform.find("a:ext", NS).set("cx", str(new_width))
        transform.find("a:ext", NS).set("cy", str(new_height))
        crop = node.find("p:blipFill/a:srcRect", NS)
        if crop is not None:
            crop.getparent().remove(crop)
    substitutions = {
        "Inference time 4.75": f"Modeled time {off_time}",
        "Inference time 2.25": f"Modeled time {on_time}",
        "Modeled time 4.75": f"Modeled time {off_time}",
        "Modeled time 2.25": f"Modeled time {on_time}",
        "4.75": off_time,
        "2.25": on_time,
        "capacity of link e": "payloads per interval",
        "worst inference time": "worst modeled time",
        "4   Inference-time lower bound": "4   Modeled inference time",
    }
    for root in roots.values():
        for text in root.findall(".//a:t", NS):
            text.text = substitutions.get(text.text, text.text).replace(
                "Inference time ", "Modeled time "
            )
    for shape_id, value in ((15, off_time), (19, on_time)):
        shape = roots[7].xpath(
            f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{shape_id}"]]', namespaces=NS
        )[0]
        runs = shape.findall(".//a:t", NS)
        for run in runs:
            run.text = ""
        runs[0].text = f"Modeled time {value}"
    append_shape(
        roots[7],
        textbox(
            "Bandwidths: DEMI, IWQoS 2025, Sec. V-A (100 / 50 Gb/s). "
            "Reduced illustrative topology; not a measured latency prediction.",
            3.35,
            7.13,
            9.4,
            0.31,
            10,
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "inputs.md"
        output = Path(directory) / "inputs.pptx"
        source.write_text(
            "$$C_{\\mathrm{local}}="
            f"{problem['network']['local_capacity']:g},"
            "\\quad C_{\\mathrm{network}}="
            f"{problem['network']['network_capacity']:g},"
            f"\\quad \\rho={problem['network']['inference_traffic']:g}"
            "\\;\\text{(assumed)}$$\n"
        )
        subprocess.run(["pandoc", str(source), "-o", str(output)], check=True)
        with zipfile.ZipFile(output) as archive:
            equation = etree.fromstring(archive.read("ppt/slides/slide1.xml"))
        math = equation.find(".//a14:m", NS)
        if math is None:
            raise RuntimeError("Editable Office Math was not generated")
        shape = textbox("", 4.3, 6.65, 8.1, 0.35, 15)
        paragraph = shape.find("p:txBody/a:p", NS)
        paragraph.insert(0, copy.deepcopy(math))
        append_shape(roots[8], shape)
    append_shape(
        roots[8],
        textbox(
            "Capacities: DEMI (2025). Bottleneck: Varys (2014). "
            "Pipelining: Le Boudec & Thiran (2001).\n"
            "Linear slowdown and 60% foreground share are assumptions; "
            "not measured or proven latency bounds.",
            0.65,
            7.08,
            12.0,
            0.34,
            10,
        ),
    )
    for page, root in roots.items():
        data[f"ppt/slides/slide{page}.xml"] = etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )
    temporary = DECK.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(temporary, "w") as archive:
        for entry in entries:
            archive.writestr(entry, data[entry.filename])
    temporary.replace(DECK)
    print(DECK)


if __name__ == "__main__":
    main()
