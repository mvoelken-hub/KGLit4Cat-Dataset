from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(r"C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)")
SOURCE = ROOT / "docs" / "presentation" / "Final_v1.pptx"
OUTPUT = ROOT / "docs" / "presentation" / "Final_v1_gui_slide.pptx"
INSPECTOR_SCREENSHOT = Path(
    r"C:\Users\simcl\AppData\Local\Temp\codex-presentations\manual-rdf-editor-showcase\tmp\assets\rdf-editor-live.png"
)


GREEN = "94C11F"
DARK_GREEN = "5C8618"
TEXT = "222222"
MUTED = "5E6470"
LINE = "D7DCE3"
PANEL = "F7F9FB"
ORANGE = "D96B00"
ORANGE_PALE = "FFF2E4"
RED = "B42318"
RED_PALE = "FFF0EF"
BLUE = "1F6FA8"
LIGHT_GREEN = "EEF7DF"


def rgb(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value)


def remove_shape(shape) -> None:
    element = shape._element
    element.getparent().remove(element)


def add_text(
    slide,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float,
    color: str = TEXT,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    font: str = "Arial",
    margin: float = 0.0,
) :
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    text_frame = shape.text_frame
    text_frame.clear()
    text_frame.word_wrap = True
    text_frame.margin_left = Inches(margin)
    text_frame.margin_right = Inches(margin)
    text_frame.margin_top = Inches(margin)
    text_frame.margin_bottom = Inches(margin)
    text_frame.vertical_anchor = valign
    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return shape


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "FFFFFF",
    line: str = LINE,
    radius: bool = False,
    line_width: float = 0.75,
):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(line_width)
    return shape


def add_tag(slide, label: str, x: float, y: float, width: float, *, fill: str, color: str) -> None:
    add_box(slide, x, y, width, 0.28, fill=fill, line=color, radius=True, line_width=0.8)
    add_text(
        slide,
        label,
        x,
        y + 0.035,
        width,
        0.18,
        size=7.6,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER,
    )


def add_tree_row(slide, label: str, x: float, y: float, *, indent: float = 0.0, selected: bool = False) -> None:
    if selected:
        add_box(slide, x, y - 0.025, 1.66, 0.265, fill=ORANGE_PALE, line="EEC38D", radius=True, line_width=0.5)
    prefix = "› " if indent else "⌄ "
    add_text(
        slide,
        f"{prefix}{label}",
        x + indent,
        y,
        1.62 - indent,
        0.20,
        size=6.8,
        color=ORANGE if selected else "3C4652",
        bold=selected,
    )


def main() -> None:
    if not INSPECTOR_SCREENSHOT.exists():
        raise FileNotFoundError(INSPECTOR_SCREENSHOT)
    shutil.copy2(SOURCE, OUTPUT)
    deck = Presentation(OUTPUT)
    slide = deck.slides[12]

    # Preserve the title explicitly. python-pptx otherwise loses the theme style
    # of this particular placeholder during its round-trip through the template.
    title = slide.shapes[0]
    title.text_frame.clear()
    title.text_frame.margin_left = Inches(0)
    title.text_frame.margin_top = Inches(0)
    title.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    title_paragraph = title.text_frame.paragraphs[0]
    title_run = title_paragraph.add_run()
    title_run.text = "Graphical user interface"
    title_run.font.name = "Arial"
    title_run.font.size = Pt(31)
    title_run.font.bold = True
    title_run.font.color.rgb = rgb(GREEN)

    # The template has a blank, malformed bullet placeholder below the title.
    for shape in list(slide.shapes):
        if (
            getattr(shape, "has_text_frame", False)
            and shape.top > Inches(2.0)
            and shape.top < Inches(2.2)
            and shape.left < Inches(1.0)
            and shape.width > Inches(5.0)
        ):
            remove_shape(shape)

    # Subtitle: the slide's two claims, before the UI examples.
    add_text(
        slide,
        "Inspect every workflow artefact - then correct the final RDF graph in context.",
        0.38,
        2.02,
        12.25,
        0.28,
        size=14,
        color=MUTED,
    )

    # Left: inspection view.
    add_tag(slide, "1  INSPECT", 0.44, 2.34, 1.12, fill=LIGHT_GREEN, color=DARK_GREEN)
    add_text(slide, "Flagged requirement with its evidence", 0.44, 2.66, 5.55, 0.28, size=14, bold=True)
    add_box(slide, 0.44, 2.98, 5.45, 3.22, fill="FFFFFF", line=LINE, radius=True, line_width=0.9)
    slide.shapes.add_picture(
        str(INSPECTOR_SCREENSHOT), Inches(0.45), Inches(2.99), width=Inches(5.43), height=Inches(3.20)
    )
    # A deliberately illustrative flag makes the inspected field concrete while retaining the actual UI view.
    add_box(slide, 0.70, 5.49, 4.40, 0.49, fill="FFFFFF", line="E7B56D", radius=True, line_width=0.85)
    add_tag(slide, "SEMANTIC REQUIREMENT", 0.82, 5.60, 1.55, fill=ORANGE_PALE, color=ORANGE)
    add_text(slide, "ACQT0 mislabelled as a duration", 2.52, 5.58, 2.32, 0.13, size=7.5, color=TEXT, bold=True)
    add_text(slide, "Requires pulse-program context", 2.52, 5.79, 2.10, 0.12, size=7.0, color=MUTED)

    # Bridge: the issue card navigates into a concrete RDF node.
    add_text(slide, "open field", 6.05, 3.98, 1.20, 0.20, size=8.5, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    arrow = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW, Inches(6.08), Inches(4.22), Inches(1.08), Inches(0.35))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = rgb(BLUE)
    arrow.line.color.rgb = rgb(BLUE)
    add_text(slide, "flag  →  RDF node", 6.00, 4.65, 1.22, 0.20, size=7.4, color=MUTED, align=PP_ALIGN.CENTER)

    # Right: explorer/editor view (illustrative UI mock, designed around the live editor's visual language).
    add_tag(slide, "2  EDIT", 7.44, 2.34, 0.88, fill="EEF5FB", color=BLUE)
    add_text(slide, "Navigate to the exact field and correct it", 7.44, 2.66, 5.10, 0.28, size=14, bold=True)
    add_box(slide, 7.44, 2.98, 5.45, 3.22, fill="FFFFFF", line=LINE, radius=True, line_width=0.9)
    add_box(slide, 7.45, 2.99, 5.43, 0.36, fill="F2F4F6", line="F2F4F6", radius=True, line_width=0)
    add_text(slide, "Curated document", 7.62, 3.095, 1.85, 0.15, size=8.8, bold=True)
    add_text(slide, "RDF explorer", 11.55, 3.095, 1.10, 0.15, size=7.6, color=MUTED, align=PP_ALIGN.RIGHT)
    add_box(slide, 7.58, 3.48, 1.78, 2.48, fill=PANEL, line="E0E4EA", radius=True, line_width=0.65)
    add_text(slide, "DATASET", 7.75, 3.64, 1.25, 0.16, size=7.4, color=MUTED, bold=True)
    add_tree_row(slide, "Dataset (8)", 7.72, 3.91)
    add_tree_row(slide, "was_generated_by", 7.72, 4.17, indent=0.13)
    add_tree_row(slide, "NMR acquisition", 7.72, 4.43, indent=0.27)
    add_tree_row(slide, "quantitative attributes", 7.72, 4.69, indent=0.40)
    add_tree_row(slide, "ACQT0  -2.52 µs", 7.72, 4.96, indent=0.54, selected=True)
    add_text(slide, "Selected field", 7.89, 5.36, 1.15, 0.15, size=7.2, color=ORANGE, bold=True)
    add_text(slide, "Evidence stays linked", 7.89, 5.61, 1.25, 0.16, size=7.0, color=MUTED)

    add_box(slide, 9.52, 3.48, 3.10, 2.48, fill="FFFFFF", line="E0E4EA", radius=True, line_width=0.65)
    add_text(slide, "QUANTITATIVE ATTRIBUTE", 9.72, 3.64, 2.25, 0.16, size=7.4, color=MUTED, bold=True)
    add_box(slide, 9.70, 3.91, 2.73, 0.43, fill=ORANGE_PALE, line="E7B56D", radius=True, line_width=0.65)
    add_text(slide, "Flagged: quantity meaning needs context", 9.84, 4.045, 2.32, 0.13, size=7.4, color=ORANGE, bold=True)
    add_text(slide, "Value", 9.72, 4.52, 0.55, 0.14, size=7.2, color=MUTED)
    add_text(slide, "−2.52 µs", 10.54, 4.52, 1.02, 0.14, size=8.5, color=TEXT, bold=True)
    add_text(slide, "Generated label", 9.72, 4.78, 0.93, 0.14, size=7.2, color=MUTED)
    add_box(slide, 10.54, 4.73, 1.60, 0.24, fill=RED_PALE, line="E9B0AA", radius=True, line_width=0.55)
    add_text(slide, "acquisition time", 10.66, 4.795, 1.26, 0.12, size=7.4, color=RED, bold=True)
    add_text(slide, "Expert edit", 9.72, 5.08, 0.80, 0.14, size=7.2, color=MUTED)
    add_box(slide, 10.54, 5.03, 1.75, 0.24, fill=LIGHT_GREEN, line="AACF72", radius=True, line_width=0.55)
    add_text(slide, "acquisition-time offset", 10.66, 5.095, 1.45, 0.12, size=6.8, color=DARK_GREEN, bold=True)
    add_text(slide, "Evidence", 9.72, 5.40, 0.58, 0.14, size=7.2, color=MUTED)
    add_text(slide, "acqt0 = -p1 × 0.66 / π", 10.54, 5.40, 1.84, 0.14, size=7.2, color=TEXT, font="Courier New")
    add_box(slide, 11.40, 5.63, 0.89, 0.20, fill=GREEN, line=GREEN, radius=True, line_width=0)
    add_text(slide, "Save", 11.40, 5.675, 0.89, 0.10, size=6.9, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER)

    # Two concise takeaways directly map to the requirements for this slide.
    add_box(slide, 0.44, 6.34, 5.45, 0.32, fill="F7FAF0", line="CDE2A5", radius=True, line_width=0.7)
    add_text(slide, "Every stage remains inspectable: evidence, draft, checks, and repair actions.", 0.64, 6.43, 5.05, 0.12, size=8.0, color=DARK_GREEN, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 7.44, 6.34, 5.45, 0.32, fill="F2F7FB", line="BCD5E8", radius=True, line_width=0.7)
    add_text(slide, "Context resolves ambiguity: the expert edits the final graph, not only a text value.", 7.64, 6.43, 5.05, 0.12, size=8.0, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

    deck.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
