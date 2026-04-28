from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/vansharora665/BTP_GP")
PRESENTATION_DIR = ROOT / "presentations"
ASSET_DIR = PRESENTATION_DIR / "presentation_assets_v4"
PPTX_TEMPLATE = PRESENTATION_DIR / "Urban_Feature_Extraction_Pipeline_Presentation_2026_03_23_v2.pptx"

JNPA_V1_SUMMARY = ROOT / "outputs/jnpa_training_run/run_summary.json"
JNPA_V2_SUMMARY = ROOT / "outputs/jnpa_patch_classifier_v2/run_summary_v2.json"
JNPA_DENSE_SUMMARY = ROOT / "outputs/jnpa_dense_segmentation_v128/run_summary_dense_v128.json"
CARTOSAT_V2_SUMMARY = ROOT / "outputs/cartosat_patch_classifier_v2/run_summary_v2.json"
CARTOSAT_V3_SUMMARY = ROOT / "outputs/cartosat_patch_classifier_v3_4class/run_summary_v3.json"
CARTOSAT_DENSE_SUMMARY = ROOT / "outputs/cartosat_dense_segmentation_v128/run_summary_dense_v128.json"

JNPA_PLATFORM_RESULT = ROOT / "urban_feature_platform/runtime/results/20260324_025516_d4571318"
CARTOSAT_PLATFORM_RESULT = ROOT / "urban_feature_platform/runtime/results/20260324_030100_64660803"

SLIDE_W = 1920
SLIDE_H = 1080
PPTX_CX = 12192000
PPTX_CY = 6858000
FORCE_WIDESCREEN_PPTX = os.environ.get("FORCE_WIDESCREEN_PPTX", "1") == "1"

BG = (245, 241, 235)
NAVY = (16, 36, 58)
TEAL = (36, 122, 142)
GOLD = (206, 154, 78)
SLATE = (80, 92, 108)
INK = (28, 30, 34)
MUTED = (92, 98, 110)
WHITE = (255, 255, 255)
CARD = (253, 252, 249)
LINE = (224, 217, 206)
GREEN = (43, 138, 82)
RED = (190, 59, 59)
BLUE = (48, 99, 179)
ORANGE = (227, 132, 56)
SOFT_BLUE = (223, 236, 248)
SOFT_GOLD = (247, 236, 215)
SOFT_GREEN = (226, 243, 232)
SOFT_RED = (247, 227, 227)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Avenir Next Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_canvas() -> Image.Image:
    image = Image.new("RGB", (SLIDE_W, SLIDE_H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SLIDE_W, 150), fill=NAVY)
    draw.rectangle((0, 144, SLIDE_W, 150), fill=GOLD)
    draw.ellipse((SLIDE_W - 320, -110, SLIDE_W + 120, 210), fill=(24, 58, 92))
    draw.ellipse((SLIDE_W - 200, -160, SLIDE_W + 160, 180), fill=(39, 85, 118))
    return image


def add_header(image: Image.Image, title: str, subtitle: str | None = None) -> None:
    draw = ImageDraw.Draw(image)
    title_font = load_font(40, bold=True)
    subtitle_font = load_font(22)
    draw.text((60, 42), title, font=title_font, fill=WHITE)
    if subtitle:
        draw.text((60, 98), subtitle, font=subtitle_font, fill=(220, 228, 238))


def add_footer_note(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    font = load_font(16)
    draw.text((60, 1040), text, font=font, fill=MUTED)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if text_size(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 8,
) -> int:
    x0, y0, x1, _ = box
    max_width = x1 - x0
    y = y0
    for line in wrap_text(draw, text, font, max_width):
        if not line:
            y += font.size // 2
            continue
        draw.text((x0, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def wrapped_text_height(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_gap: int = 8,
) -> int:
    total = 0
    for line in wrap_text(draw, text, font, max_width):
        if not line:
            total += font.size // 2
        else:
            total += font.size + line_gap
    return total


def draw_bullet_list(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    bullet_color: tuple[int, int, int],
) -> int:
    x0, y0, x1, _ = box
    y = y0
    bullet_radius = 5
    indent = 28
    for item in items:
        lines = wrap_text(draw, item, font, x1 - x0 - indent)
        draw.ellipse((x0, y + 10, x0 + bullet_radius * 2, y + 10 + bullet_radius * 2), fill=bullet_color)
        for line_index, line in enumerate(lines):
            draw.text((x0 + indent, y + line_index * (font.size + 7)), line, font=font, fill=fill)
        y += len(lines) * (font.size + 7) + 10
    return y


def bullet_list_height(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> int:
    total = 0
    indent = 28
    for item in items:
        lines = wrap_text(draw, item, font, max_width - indent)
        total += len(lines) * (font.size + 7) + 10
    return total


def fit_body_font(
    draw: ImageDraw.ImageDraw,
    *,
    body_lines: list[str] | None,
    body_text: str | None,
    max_width: int,
    max_height: int,
    start_size: int = 22,
    min_size: int = 17,
) -> ImageFont.ImageFont:
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size)
        if body_lines:
            height = bullet_list_height(draw, body_lines, font, max_width)
        else:
            height = wrapped_text_height(draw, body_text or "", font, max_width, line_gap=7)
        if height <= max_height:
            return font
    return load_font(min_size)


def draw_card(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    body_lines: list[str] | None = None,
    body_text: str | None = None,
    accent: tuple[int, int, int] = TEAL,
    fill: tuple[int, int, int] = WHITE,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=LINE, width=2)
    draw.rounded_rectangle((x0 + 20, y0 + 22, x0 + 34, y0 + 142), radius=8, fill=accent)
    title_font = load_font(28, bold=True)
    draw.text((x0 + 58, y0 + 26), title, font=title_font, fill=INK)
    max_body_width = x1 - x0 - 86
    max_body_height = y1 - y0 - 110
    body_font = fit_body_font(
        draw,
        body_lines=body_lines,
        body_text=body_text,
        max_width=max_body_width,
        max_height=max_body_height,
    )
    if body_lines:
        draw_bullet_list(
            draw,
            body_lines,
            (x0 + 58, y0 + 82, x1 - 28, y1 - 20),
            body_font,
            MUTED,
            accent,
        )
    elif body_text:
        draw_wrapped_text(draw, body_text, (x0 + 58, y0 + 84, x1 - 28, y1 - 20), body_font, MUTED, line_gap=7)


def metric_chip(image: Image.Image, position: tuple[int, int], label: str, value: str, accent: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    x, y = position
    box = (x, y, x + 250, y + 92)
    draw.rounded_rectangle(box, radius=24, fill=WHITE, outline=LINE, width=2)
    draw.text((x + 18, y + 16), label, font=load_font(18, bold=True), fill=MUTED)
    draw.text((x + 18, y + 42), value, font=load_font(28, bold=True), fill=accent)


def paste_contain(base: Image.Image, source: Image.Image, box: tuple[int, int, int, int], border: bool = True) -> None:
    x0, y0, x1, y1 = box
    if border:
        draw = ImageDraw.Draw(base)
        draw.rounded_rectangle(box, radius=24, fill=WHITE, outline=LINE, width=2)
        x0 += 16
        y0 += 16
        x1 -= 16
        y1 -= 16
    target_w = x1 - x0
    target_h = y1 - y0
    scale = min(target_w / source.width, target_h / source.height)
    resized = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.LANCZOS)
    paste_x = x0 + (target_w - resized.width) // 2
    paste_y = y0 + (target_h - resized.height) // 2
    base.paste(resized, (paste_x, paste_y))


def paste_titled_image_card(
    base: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str | None = None,
) -> None:
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=WHITE, outline=LINE, width=2)
    title_font = load_font(22, bold=True)
    subtitle_font = load_font(16)
    header_height = 72 if subtitle else 58
    draw.text((x0 + 20, y0 + 14), title, font=title_font, fill=INK)
    if subtitle:
        draw.text((x0 + 20, y0 + 42), subtitle, font=subtitle_font, fill=MUTED)
    inner_box = (x0 + 16, y0 + header_height, x1 - 16, y1 - 16)
    paste_contain(base, source, inner_box, border=False)


def load_preview(image_path: Path, out_path: Path, width: int = 1000) -> Path:
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)
        valid_mask = np.ones_like(image, dtype=bool)
        if src.nodata is not None:
            valid_mask &= np.not_equal(image, src.nodata)
        values = image[valid_mask]
        lo, hi = np.percentile(values, [1, 99])
        image = np.clip(image, lo, hi)
        image = (image - lo) / max(hi - lo, 1e-6)
        image[~valid_mask] = 0.0
    height = int(image.shape[0] * width / image.shape[1])
    preview = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height))
    preview.convert("RGB").save(out_path)
    return out_path


def save_overall_metrics_chart(
    out_path: Path,
    labels: list[str],
    pixel_values: list[float],
    miou_values: list[float],
    title: str,
) -> Path:
    colors = ["#94a3b8", "#f97316", "#2563eb"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(x, np.array(pixel_values) * 100.0, color=colors, width=0.58)
    axes[0].set_title("Pixel Accuracy")
    axes[0].set_ylim(0, 110)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    for idx, value in enumerate(pixel_values):
        axes[0].text(idx, value * 100 + 2, f"{value * 100:.1f}", ha="center", fontsize=10)

    axes[1].bar(x, np.array(miou_values) * 100.0, color=colors, width=0.58)
    axes[1].set_title("Mean IoU")
    axes[1].set_ylim(0, 110)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    for idx, value in enumerate(miou_values):
        axes[1].text(idx, value * 100 + 2, f"{value * 100:.1f}", ha="center", fontsize=10)

    fig.suptitle(title, fontsize=16)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_class_accuracy_chart(
    out_path: Path,
    class_names: list[str],
    baseline_acc: list[float],
    hard_acc: list[float],
    soft_acc: list[float],
    title: str,
) -> Path:
    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.bar(x - width, np.array(baseline_acc) * 100.0, width=width, label="Baseline teacher", color="#94a3b8")
    ax.bar(x, np.array(hard_acc) * 100.0, width=width, label="Hard U-Net", color="#f97316")
    ax.bar(x + width, np.array(soft_acc) * 100.0, width=width, label="Soft U-Net", color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel("Class Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=18, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_legend_rows(
    image: Image.Image,
    start_xy: tuple[int, int],
    class_names: list[str],
    colors: list[tuple[int, int, int]],
    column_width: int,
    items_per_column: int,
) -> None:
    draw = ImageDraw.Draw(image)
    font = load_font(20)
    x0, y0 = start_xy
    for idx, (label, color) in enumerate(zip(class_names, colors)):
        col = idx // items_per_column
        row = idx % items_per_column
        x = x0 + col * column_width
        y = y0 + row * 34
        draw.rounded_rectangle((x, y + 3, x + 20, y + 23), radius=6, fill=color)
        draw.text((x + 32, y), label, font=font, fill=MUTED)


def make_applescript_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{escaped}\""


def generate_pptx_slide_xml(image_name: str, display_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" showMasterSp="0" showMasterPhAnim="0"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:pic><p:nvPicPr><p:cNvPr id="2" name="{display_name}" descr="{display_name}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{PPTX_CX}" cy="{PPTX_CY}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def generate_pptx_slide_rel_xml(image_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout14.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/></Relationships>"""


def build_pptx_from_template(slide_paths: list[Path], pptx_path: Path) -> None:
    with zipfile.ZipFile(PPTX_TEMPLATE) as zin, zipfile.ZipFile(pptx_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            name = info.filename
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name):
                continue
            if re.fullmatch(r"ppt/slides/_rels/slide\d+\.xml\.rels", name):
                continue
            if re.fullmatch(r"ppt/media/image\d+\.(png|jpg|jpeg)", name):
                continue
            if name in {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml"}:
                continue
            zout.writestr(name, zin.read(name))

        presentation_xml = zin.read("ppt/presentation.xml").decode("utf-8")
        slide_id_list = "".join(
            f'<p:sldId id="{256 + index}" r:id="rId{8 + index}"/>'
            for index in range(len(slide_paths))
        )
        presentation_xml = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", f"<p:sldIdLst>{slide_id_list}</p:sldIdLst>", presentation_xml, flags=re.DOTALL)
        presentation_xml = re.sub(r'<p:sldSz cx="\d+" cy="\d+"\/>', f'<p:sldSz cx="{PPTX_CX}" cy="{PPTX_CY}"/>', presentation_xml)
        zout.writestr("ppt/presentation.xml", presentation_xml)

        fixed_rels = [
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/commentAuthors", "commentAuthors.xml"),
            ("rId4", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
            ("rId5", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId6", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
            ("rId7", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster", "notesMasters/notesMaster1.xml"),
        ]
        rel_lines = [
            f'<Relationship Id="{rel_id}" Type="{rel_type}" Target="{target}"/>'
            for rel_id, rel_type, target in fixed_rels
        ]
        for index in range(len(slide_paths)):
            rel_lines.append(
                f'<Relationship Id="rId{8 + index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{index + 1}.xml"/>'
            )
        presentation_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rel_lines)
            + "</Relationships>"
        )
        zout.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml)

        content_types_xml = zin.read("[Content_Types].xml").decode("utf-8")
        content_types_xml = re.sub(
            r'<Override PartName="/ppt/slides/slide\d+\.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide\+xml"/>',
            "",
            content_types_xml,
        )
        slide_override_block = "".join(
            f'<Override PartName="/ppt/slides/slide{index + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for index in range(len(slide_paths))
        )
        insertion_point = content_types_xml.index('<Override PartName="/ppt/slideLayouts/slideLayout1.xml"')
        content_types_xml = content_types_xml[:insertion_point] + slide_override_block + content_types_xml[insertion_point:]
        zout.writestr("[Content_Types].xml", content_types_xml)

        for index, slide_path in enumerate(slide_paths, start=1):
            image_name = f"image{index}.png"
            display_name = slide_path.name
            zout.writestr(f"ppt/slides/slide{index}.xml", generate_pptx_slide_xml(image_name, display_name))
            zout.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", generate_pptx_slide_rel_xml(image_name))
            zout.writestr(f"ppt/media/{image_name}", slide_path.read_bytes())


def create_cover_slide(
    out_path: Path,
    jnpa_preview: Path,
    cart_preview: Path,
    jnpa_overlay: Path,
    cart_overlay: Path,
    jnpa_dense: dict,
    cart_dense: dict,
) -> Path:
    image = Image.new("RGB", (SLIDE_W, SLIDE_H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SLIDE_W, 1080), fill=(242, 238, 231))
    draw.rectangle((0, 0, 980, SLIDE_H), fill=NAVY)
    draw.ellipse((680, -120, 1180, 320), fill=(33, 76, 109))
    draw.ellipse((760, 40, 1180, 420), fill=(28, 95, 118))

    title_font = load_font(54, bold=True)
    subtitle_font = load_font(24)
    chip_font = load_font(21, bold=True)
    body_font = load_font(24)

    draw.text((72, 92), "Urban Feature Extraction\nfrom Panchromatic\nSatellite Imagery", font=title_font, fill=WHITE, spacing=12)
    draw.text(
        (72, 300),
        "Curated final presentation covering the full pipeline,\nmodel evolution, final dense 128x128 results,\nand the deployable interactive platform.",
        font=subtitle_font,
        fill=(219, 228, 238),
        spacing=8,
    )

    chips = [
        ("JNPA + CARTOSAT", TEAL),
        ("Dense U-Net V128", GOLD),
        ("Production Platform", GREEN),
    ]
    chip_x = 72
    for label, color in chips:
        tw, _ = text_size(draw, label, chip_font)
        draw.rounded_rectangle((chip_x, 420, chip_x + tw + 34, 462), radius=18, fill=color)
        draw.text((chip_x + 16, 430), label, font=chip_font, fill=WHITE)
        chip_x += tw + 58

    summary_box = (72, 545, 900, 900)
    draw.rounded_rectangle(summary_box, radius=32, fill=(247, 244, 238), outline=(56, 84, 112), width=2)
    draw.text((110, 580), "Final Best Models", font=load_font(30, bold=True), fill=INK)
    draw.text(
        (110, 636),
        f"JNPA Soft U-Net V128\nPixel accuracy {jnpa_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}%\n"
        f"Mean IoU {jnpa_dense['best_variant_metrics']['mean_iou'] * 100:.2f}%",
        font=body_font,
        fill=MUTED,
        spacing=10,
    )
    draw.text(
        (110, 766),
        f"CARTOSAT Hard U-Net V128\nPixel accuracy {cart_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}%\n"
        f"Mean IoU {cart_dense['best_variant_metrics']['mean_iou'] * 100:.2f}%",
        font=body_font,
        fill=MUTED,
        spacing=10,
    )

    tile_positions = [
        (1050, 90, 1430, 420),
        (1470, 90, 1850, 420),
        (1050, 470, 1430, 930),
        (1470, 470, 1850, 930),
    ]
    tile_titles = [
        ("JNPA raw scene", jnpa_preview),
        ("CARTOSAT raw scene", cart_preview),
        ("JNPA final dense output", jnpa_overlay),
        ("CARTOSAT final dense output", cart_overlay),
    ]
    for box, (label, path) in zip(tile_positions, tile_titles):
        paste_titled_image_card(image, Image.open(path).convert("RGB"), box, label)

    draw.text((72, 972), "Professor presentation version | March 2026 | All metrics below are from saved validation summaries inside this repository.", font=load_font(18), fill=(214, 223, 233))
    image.save(out_path)
    return out_path


def create_dataset_slide(out_path: Path, jnpa_preview: Path, cart_preview: Path) -> Path:
    image = make_canvas()
    add_header(image, "Project Context and Datasets", "Problem definition, data characteristics, and final class structure")
    draw = ImageDraw.Draw(image)

    draw_card(
        image,
        (60, 205, 900, 470),
        "Project Objective",
        body_lines=[
            "Extract urban and environmental features from high-resolution monochromatic satellite imagery.",
            "Produce segmented land-use maps for water, industrial/port zones, vegetation, bare land, and urban structures.",
            "Build an end-to-end workflow that can move from raw GeoTIFF to deployable inference outputs.",
        ],
        accent=TEAL,
    )
    draw_card(
        image,
        (980, 205, 1860, 470),
        "Why Panchromatic Imagery Is Hard",
        body_lines=[
            "Only one grayscale band is available, so spectral color cues are missing.",
            "The model must rely on intensity, texture, edges, contrast, and spatial structure.",
            "Roads, roofs, port yards, water, vegetation, and bare land can look similar without careful texture modeling.",
        ],
        accent=GOLD,
    )

    paste_titled_image_card(image, Image.open(jnpa_preview).convert("RGB"), (60, 520, 900, 850), "JNPA 2.5m PAN")
    paste_titled_image_card(image, Image.open(cart_preview).convert("RGB"), (980, 520, 1860, 850), "CARTOSAT 1m PAN")
    draw.rounded_rectangle((60, 870, 900, 980), radius=24, fill=WHITE, outline=LINE, width=2)
    draw.rounded_rectangle((980, 870, 1860, 980), radius=24, fill=WHITE, outline=LINE, width=2)

    add_legend_rows(
        image,
        (92, 902),
        ["Water Bodies", "Industrial / Port Infrastructure", "Bare Land / Soil", "Vegetation / Mangroves", "Urban Built-up"],
        [(31, 119, 180), (214, 39, 40), (140, 86, 75), (44, 160, 44), (255, 215, 0)],
        360,
        3,
    )
    add_legend_rows(
        image,
        (1012, 902),
        ["Water", "Dense Urban Built-up", "Port / Waterfront Infrastructure", "Terrain / Open Ground"],
        [(31, 119, 180), (255, 215, 0), (214, 39, 40), (140, 86, 75)],
        370,
        2,
    )

    add_footer_note(image, "Final class definitions used in the delivered models and the interactive web platform.")
    image.save(out_path)
    return out_path


def create_dataset_detail_slide(out_path: Path, jnpa_preview: Path, cart_preview: Path, jnpa_dense: dict, cart_dense: dict) -> Path:
    image = make_canvas()
    add_header(image, "Understanding the Two Datasets", "Detailed dataset profile before discussing the modeling work")

    paste_titled_image_card(image, Image.open(jnpa_preview).convert("RGB"), (60, 215, 900, 555), "JNPA scene overview")
    paste_titled_image_card(image, Image.open(cart_preview).convert("RGB"), (980, 215, 1860, 555), "CARTOSAT scene overview")

    draw_card(
        image,
        (60, 595, 900, 975),
        "JNPA Dataset Profile",
        body_lines=[
            "Sensor and scale: JNPA panchromatic imagery at 2.5 m spatial resolution.",
            f"Raster size: {jnpa_dense['dataset']['raster_metadata']['width']} x {jnpa_dense['dataset']['raster_metadata']['height']} pixels | dtype {jnpa_dense['dataset']['raster_metadata']['dtype']}.",
            f"Normalization range used in the pipeline: {jnpa_dense['dataset']['raster_metadata']['normalization_low']} to {jnpa_dense['dataset']['raster_metadata']['normalization_high']}.",
            "Main land-cover patterns: harbor water, dense industrial/port yards, mangroves, bare coastal soil, and urban built-up blocks.",
            f"Final classes: {len(jnpa_dense['dataset']['class_names'])} semantic categories.",
            f"Teacher seed patches: {jnpa_dense['dataset']['teacher_seed_patch_count']} | Dense 128x128 dataset patches: {jnpa_dense['dataset']['dense_dataset_patch_count']}.",
            "Final selected production model: soft-label dense U-Net V128.",
        ],
        accent=TEAL,
        fill=(247, 252, 250),
    )
    draw_card(
        image,
        (980, 595, 1860, 975),
        "CARTOSAT Dataset Profile",
        body_lines=[
            "Sensor and scale: CARTOSAT-1 panchromatic imagery at 1 m spatial resolution.",
            f"Raster size: {cart_dense['dataset']['raster_metadata']['width']} x {cart_dense['dataset']['raster_metadata']['height']} pixels | dtype {cart_dense['dataset']['raster_metadata']['dtype']}.",
            f"Normalization range used in the pipeline: {cart_dense['dataset']['raster_metadata']['normalization_low']} to {cart_dense['dataset']['raster_metadata']['normalization_high']}.",
            "Main land-cover patterns: open water, dense urban fabric, engineered port infrastructure, and exposed open terrain.",
            f"Final classes: {len(cart_dense['dataset']['class_names'])} semantic categories.",
            f"Teacher seed patches: {cart_dense['dataset']['teacher_seed_patch_count']} | Dense 128x128 dataset patches: {cart_dense['dataset']['dense_dataset_patch_count']}.",
            "Final selected production model: hard-mask dense U-Net V128.",
        ],
        accent=GOLD,
        fill=(255, 250, 241),
    )

    add_footer_note(image, "Both datasets are single-band panchromatic scenes, so all discrimination comes from grayscale texture, contrast, and spatial structure instead of color.")
    image.save(out_path)
    return out_path


def create_dataset_process_slide(
    out_path: Path,
    title: str,
    subtitle: str,
    image_cards: list[tuple[str, str | None, Path]],
    step_cards: list[tuple[str, str, tuple[int, int, int], tuple[int, int, int]]],
    footer_note: str,
) -> Path:
    image = make_canvas()
    add_header(image, title, subtitle)

    image_boxes = [
        (60, 210, 610, 620),
        (685, 210, 1235, 620),
        (1310, 210, 1860, 620),
    ]
    for box, (label, sublabel, path) in zip(image_boxes, image_cards):
        paste_titled_image_card(image, Image.open(path).convert("RGB"), box, label, sublabel)

    step_boxes = [
        (60, 690, 470, 970),
        (510, 690, 920, 970),
        (960, 690, 1370, 970),
        (1410, 690, 1860, 970),
    ]
    for box, (card_title, body_text, accent, fill) in zip(step_boxes, step_cards):
        draw_card(image, box, card_title, body_text=body_text, accent=accent, fill=fill)

    add_footer_note(image, footer_note)
    image.save(out_path)
    return out_path


def create_labeling_slide(out_path: Path, dense_preview_panel: Path) -> Path:
    image = make_canvas()
    add_header(image, "Pseudo-Labeling Strategy", "How the project moved from raw grayscale imagery to dense pixel-level supervision")
    draw = ImageDraw.Draw(image)

    steps = [
        ("1. Patch-level discovery", "Split the raster into 256x256 teacher patches, remove low-information tiles, and cluster them using intensity and texture features."),
        ("2. Human semantic mapping", "Inspect cluster contact sheets and assign each cluster to a real-world class in code instead of drawing masks by hand."),
        ("3. Dense supervision upgrade", "Train a pixel teacher on local grayscale/texture cues, then convert patch knowledge into hard masks and soft probability maps."),
    ]
    y = 215
    for index, (title, body) in enumerate(steps):
        accent = [TEAL, GOLD, GREEN][index]
        draw_card(image, (60, y, 850, y + 210), title, body_text=body, accent=accent)
        y += 230

    paste_titled_image_card(
        image,
        Image.open(dense_preview_panel).convert("RGB"),
        (910, 215, 1860, 720),
        "Dense pseudo-label preview samples",
    )

    draw_card(
        image,
        (910, 760, 1380, 980),
        "Hard Masks",
        body_lines=[
            "Each pixel is assigned one class value.",
            "Used as the winning production variant for CARTOSAT.",
            "Best when class boundaries are sharp and confident.",
        ],
        accent=ORANGE,
    )
    draw_card(
        image,
        (1410, 760, 1860, 980),
        "Soft Labels",
        body_lines=[
            "Each pixel carries class probabilities instead of a single hard label.",
            "Used as the winning production variant for JNPA.",
            "Best when mixed boundaries or uncertain transitions exist.",
        ],
        accent=BLUE,
    )

    add_footer_note(image, "Important note: this is still pseudo-label supervision, so reported metrics are not manual ground-truth accuracy.")
    image.save(out_path)
    return out_path


def create_evolution_slide(
    out_path: Path,
    jnpa_v1: dict,
    jnpa_v2: dict,
    jnpa_dense: dict,
    cart_v3: dict,
    cart_dense: dict,
) -> Path:
    image = make_canvas()
    add_header(image, "Model Evolution and Key Design Decisions", "What changed from the first attempt to the final deployed solution")
    draw = ImageDraw.Draw(image)

    boxes = [
        (60, 250, 610, 900),
        (685, 250, 1235, 900),
        (1310, 250, 1860, 900),
    ]

    draw_card(
        image,
        boxes[0],
        "Version 1 | Direct U-Net",
        body_lines=[
            "Patch size: 256x256",
            "Pseudo-mask per patch with weak dense supervision",
            f"JNPA pixel accuracy: {jnpa_v1['final_val_metrics']['pixel_accuracy'] * 100:.2f}%",
            f"JNPA mean IoU: {jnpa_v1['final_val_metrics']['mean_iou'] * 100:.2f}%",
            "Main issue: labels were patch-level, but the model was asked to learn pixel-level boundaries.",
        ],
        accent=SLATE,
        fill=(248, 248, 248),
    )
    draw_card(
        image,
        boxes[1],
        "Version 2 | Patch Classifier",
        body_lines=[
            "Random-forest classifier on PCA-reduced texture features",
            f"JNPA val accuracy: {jnpa_v2['metrics']['val_accuracy'] * 100:.2f}%",
            f"JNPA macro F1: {jnpa_v2['metrics']['val_macro_f1'] * 100:.2f}%",
            f"CARTOSAT 4-class val accuracy: {cart_v3['metrics']['val_accuracy'] * 100:.2f}%",
            f"CARTOSAT 4-class macro F1: {cart_v3['metrics']['val_macro_f1'] * 100:.2f}%",
            "Main benefit: model output matched patch-level pseudo labels, but dense segmentation quality was limited.",
        ],
        accent=TEAL,
        fill=(247, 252, 250),
    )
    draw_card(
        image,
        boxes[2],
        "Final Version | Dense U-Net V128",
        body_lines=[
            "128x128 student patches",
            "Hard-mask and soft-label training compared for each dataset",
            f"JNPA winner: Soft U-Net | {jnpa_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {jnpa_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mIoU",
            f"CARTOSAT winner: Hard U-Net | {cart_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {cart_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mIoU",
            "Final result: better boundary quality, better IoU, and deployment-ready dense maps.",
        ],
        accent=GOLD,
        fill=(255, 250, 241),
    )

    draw.rounded_rectangle((60, 940, 1860, 1010), radius=24, fill=SOFT_BLUE, outline=LINE, width=2)
    draw.text(
        (90, 958),
        "Core lesson: once the supervision was upgraded from one label per patch to dense pixel pseudo-labels, U-Net became the correct model again and produced the best final outputs.",
        font=load_font(24, bold=True),
        fill=INK,
    )
    add_footer_note(image, "The patch classifier remains available for comparison, but the platform production models now use the dense V128 winners.")
    image.save(out_path)
    return out_path


def create_pipeline_slide(out_path: Path) -> Path:
    image = make_canvas()
    add_header(image, "Final Dense 128x128 Training Pipeline", "End-to-end workflow used for the delivered production models")
    draw = ImageDraw.Draw(image)

    step_titles = [
        "Load PAN GeoTIFF",
        "Robust normalization",
        "256x256 teacher patches",
        "Texture features + clustering",
        "Manual cluster-to-class mapping",
        "Pixel teacher MLP",
        "Dense hard masks + soft probabilities",
        "128x128 student patches",
        "Train hard and soft U-Nets",
        "Select best model and reconstruct full map",
        "Serve in web platform",
    ]
    step_colors = [SOFT_BLUE, SOFT_GREEN, SOFT_GOLD, (237, 233, 254), SOFT_RED, SOFT_BLUE, SOFT_GREEN, SOFT_GOLD, (237, 233, 254), SOFT_RED, SOFT_BLUE]

    box_w = 300
    box_h = 112
    gap_x = 26
    gap_y = 90
    start_x = 80
    start_y = 245
    positions = []
    for row in range(2):
        for col in range(4 if row == 0 else 4):
            positions.append((start_x + col * (box_w + gap_x), start_y + row * (box_h + gap_y)))
    positions.extend([(80, 635), (406, 635), (732, 635)])

    for idx, title in enumerate(step_titles):
        x, y = positions[idx]
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=22, fill=step_colors[idx], outline=LINE, width=2)
        lines = title.split("\n") if "\n" in title else wrap_text(draw, title, load_font(24, bold=True), box_w - 34)
        for line_index, line in enumerate(lines):
            tw, th = text_size(draw, line, load_font(24, bold=True))
            draw.text((x + (box_w - tw) / 2, y + 26 + line_index * 28), line, font=load_font(24, bold=True), fill=INK)
        if idx < len(step_titles) - 1:
            if idx in {3, 7}:
                continue
            x0 = x + box_w
            x1 = positions[idx + 1][0] - 8
            mid_y = y + box_h // 2
            draw.line((x0 + 8, mid_y, x1, mid_y), fill=SLATE, width=5)
            draw.polygon([(x1, mid_y), (x1 - 16, mid_y - 10), (x1 - 16, mid_y + 10)], fill=SLATE)

    draw.line((1332, 301, 1332, 430), fill=SLATE, width=5)
    draw.polygon([(1332, 430), (1322, 412), (1342, 412)], fill=SLATE)
    draw.line((80 + box_w // 2, 747, 1680, 747), fill=(0, 0, 0, 0), width=0)

    draw_card(
        image,
        (1380, 245, 1860, 515),
        "Training Settings",
        body_lines=[
            "Student patch size: 128x128",
            "Base channels: 16",
            "Batch size: 24",
            "Learning rate: 0.0008",
            "Compare hard-mask and soft-label losses, then promote only if mean IoU improves.",
        ],
        accent=TEAL,
    )
    draw_card(
        image,
        (1085, 635, 1860, 960),
        "Why This Final Pipeline Worked Better",
        body_lines=[
            "It preserves the useful cluster-based pseudo-labeling idea, but no longer assumes one label for an entire patch.",
            "The final U-Net sees finer 128x128 patches and learns local boundaries from dense pseudo masks.",
            "Soft labels help in mixed transition regions, while hard masks remain strong for clean engineered surfaces.",
        ],
        accent=GOLD,
    )

    add_footer_note(image, "Promotion rule used in the code: replace a production model only if the new dense version improves mean IoU by at least 0.03.")
    image.save(out_path)
    return out_path


def create_quant_slide(
    out_path: Path,
    title: str,
    subtitle: str,
    overall_chart: Path,
    class_chart: Path,
    cards: list[tuple[str, str, tuple[int, int, int], tuple[int, int, int]]],
    note: str,
) -> Path:
    image = make_canvas()
    add_header(image, title, subtitle)
    paste_titled_image_card(image, Image.open(overall_chart).convert("RGB"), (60, 210, 930, 640), "Overall validation comparison")
    paste_titled_image_card(image, Image.open(class_chart).convert("RGB"), (990, 210, 1860, 640), "Class-wise validation accuracy")

    box_positions = [
        (60, 690, 470, 970),
        (510, 690, 920, 970),
        (960, 690, 1370, 970),
        (1410, 690, 1860, 970),
    ]
    for (header, body, accent, fill), box in zip(cards, box_positions):
        draw_card(image, box, header, body_text=body, accent=accent, fill=fill)

    add_footer_note(image, note)
    image.save(out_path)
    return out_path


def create_map_comparison_slide(
    out_path: Path,
    title: str,
    subtitle: str,
    baseline_overlay: Path,
    final_overlay: Path,
    baseline_title: str,
    final_title: str,
    baseline_metrics: tuple[float, float],
    final_metrics: tuple[float, float],
    improvement_text: str,
    takeaway_lines: list[str],
) -> Path:
    image = make_canvas()
    add_header(image, title, subtitle)
    draw = ImageDraw.Draw(image)

    left_box = (70, 225, 920, 760)
    right_box = (1000, 225, 1850, 760)
    paste_titled_image_card(image, Image.open(baseline_overlay).convert("RGB"), left_box, baseline_title)
    paste_titled_image_card(image, Image.open(final_overlay).convert("RGB"), right_box, final_title)
    metric_chip(image, (100, 680), "Pixel accuracy", f"{baseline_metrics[0] * 100:.2f}%", SLATE)
    metric_chip(image, (370, 680), "Mean IoU", f"{baseline_metrics[1] * 100:.2f}%", SLATE)
    metric_chip(image, (1030, 680), "Pixel accuracy", f"{final_metrics[0] * 100:.2f}%", BLUE)
    metric_chip(image, (1300, 680), "Mean IoU", f"{final_metrics[1] * 100:.2f}%", BLUE)

    draw_card(
        image,
        (70, 820, 680, 980),
        "Observed Improvement",
        body_text=improvement_text,
        accent=GOLD,
        fill=(255, 250, 241),
    )
    draw_card(
        image,
        (730, 820, 1850, 980),
        "Interpretation",
        body_lines=takeaway_lines,
        accent=TEAL,
        fill=(247, 252, 250),
    )
    add_footer_note(image, "Map comparison uses the dense teacher baseline versus the final promoted dense model for the same dataset.")
    image.save(out_path)
    return out_path


def create_platform_slide(
    out_path: Path,
    jnpa_result_dir: Path,
    cart_result_dir: Path,
) -> Path:
    image = make_canvas()
    add_header(image, "Production-ready Interactive Platform", "What the final web system does for non-technical users")
    draw = ImageDraw.Draw(image)

    draw_card(
        image,
        (60, 205, 760, 520),
        "Platform Capabilities",
        body_lines=[
            "Upload common image formats: png, jpg, jpeg, tif, tiff, bmp, webp, gif.",
            "Auto mode selects the correct dataset family; manual mode lets users compare models.",
            "Large GeoTIFF scenes up to roughly 500 MB are handled with streaming raster inference.",
            "Outputs include marked preview, segmentation map, class percentages, and georeferenced label GeoTIFF.",
            "Registry-driven backend makes retraining or adding a new dataset easy without rewriting the UI.",
        ],
        accent=TEAL,
    )

    draw_card(
        image,
        (60, 560, 760, 980),
        "Operational Flow",
        body_lines=[
            "1. User uploads image",
            "2. Platform normalizes and tiles scene",
            "3. Auto/manual model selection",
            "4. Dense inference and class percentage calculation",
            "5. Preview images and download files are generated",
            "6. Results can be demonstrated locally over the same Wi-Fi network",
        ],
        accent=GOLD,
    )

    preview_boxes = [
        (830, 205, 1310, 565),
        (1360, 205, 1840, 565),
        (830, 610, 1310, 970),
        (1360, 610, 1840, 970),
    ]
    previews = [
        ("JNPA marked preview", jnpa_result_dir / "marked_preview.png"),
        ("JNPA segmentation preview", jnpa_result_dir / "segmentation_preview.png"),
        ("CARTOSAT marked preview", cart_result_dir / "marked_preview.png"),
        ("CARTOSAT segmentation preview", cart_result_dir / "segmentation_preview.png"),
    ]
    for box, (label, path) in zip(preview_boxes, previews):
        paste_titled_image_card(image, Image.open(path).convert("RGB"), box, label)

    add_footer_note(image, "The deployed platform uses the final promoted dense models: jnpa_dense_v128_prod and cartosat_dense_v128_prod.")
    image.save(out_path)
    return out_path


def create_conclusion_slide(out_path: Path, jnpa_dense: dict, cart_dense: dict) -> Path:
    image = make_canvas()
    add_header(image, "Final Conclusions", "Key outcomes, practical deliverables, and honest reporting notes")
    draw = ImageDraw.Draw(image)

    draw_card(
        image,
        (60, 220, 600, 520),
        "Main Technical Outcome",
        body_lines=[
            "Dense pseudo-labeling plus 128x128 U-Net training produced the strongest final results.",
            f"JNPA final: {jnpa_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {jnpa_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mean IoU",
            f"CARTOSAT final: {cart_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {cart_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mean IoU",
        ],
        accent=BLUE,
        fill=(244, 248, 255),
    )
    draw_card(
        image,
        (690, 220, 1230, 520),
        "Delivered System",
        body_lines=[
            "Training pipelines for clustering, dense pseudo-label generation, and U-Net comparison",
            "Versioned model artifacts for JNPA and CARTOSAT",
            "Working interactive platform for local demos and large-raster inference",
            "Documentation for retraining, model registry management, and deployment",
        ],
        accent=TEAL,
        fill=(247, 252, 250),
    )
    draw_card(
        image,
        (1320, 220, 1860, 520),
        "Reporting Note",
        body_lines=[
            "The validation metrics are legitimate for the pseudo-label setup used in this project.",
            "They should not be claimed as manual ground-truth segmentation accuracy.",
            "This is an honest and important limitation to state during the presentation.",
        ],
        accent=RED,
        fill=(252, 244, 244),
    )

    draw_card(
        image,
        (60, 590, 880, 960),
        "Future Improvements",
        body_lines=[
            "Add manually annotated ground-truth masks for a smaller benchmark set.",
            "Expand to more geographies and more panchromatic sensors.",
            "Introduce uncertainty maps and active-learning loops for manual correction.",
            "Deploy the platform on a cloud or lab server for multi-user access.",
        ],
        accent=GOLD,
        fill=(255, 250, 241),
    )

    draw.rounded_rectangle((950, 610, 1860, 940), radius=36, fill=NAVY)
    draw.text((1010, 680), "Thank You", font=load_font(56, bold=True), fill=WHITE)
    draw.text(
        (1010, 770),
        "The final repository now contains:\n"
        "- trained dense models\n"
        "- result maps and comparisons\n"
        "- the interactive platform\n"
        "- presentation-ready project outputs",
        font=load_font(28),
        fill=(224, 233, 242),
        spacing=12,
    )
    add_footer_note(image, "Presentation generated directly from repository artifacts to keep the story consistent with the final code and results.")
    image.save(out_path)
    return out_path


def build_presentation() -> tuple[Path | None, Path]:
    ensure_dir(PRESENTATION_DIR)
    ensure_dir(ASSET_DIR)

    jnpa_v1 = load_json(JNPA_V1_SUMMARY)
    jnpa_v2 = load_json(JNPA_V2_SUMMARY)
    jnpa_dense = load_json(JNPA_DENSE_SUMMARY)
    cart_v2 = load_json(CARTOSAT_V2_SUMMARY)
    cart_v3 = load_json(CARTOSAT_V3_SUMMARY)
    cart_dense = load_json(CARTOSAT_DENSE_SUMMARY)

    jnpa_preview = load_preview(ROOT / "JNPA/JNPA_2_5.tif", ASSET_DIR / "jnpa_preview_v3.png", width=960)
    cart_preview = load_preview(ROOT / "Monocromatic/CARTOSAT_1M_PAN.tif", ASSET_DIR / "cartosat_preview_v3.png", width=960)

    jnpa_overall = save_overall_metrics_chart(
        ASSET_DIR / "jnpa_overall_chart_v3.png",
        ["Teacher baseline", "Hard U-Net", "Soft U-Net"],
        [
            jnpa_dense["teacher_baseline_metrics"]["pixel_accuracy"],
            jnpa_dense["hard_unet"]["final_val_metrics"]["pixel_accuracy"],
            jnpa_dense["soft_unet"]["final_val_metrics"]["pixel_accuracy"],
        ],
        [
            jnpa_dense["teacher_baseline_metrics"]["mean_iou"],
            jnpa_dense["hard_unet"]["final_val_metrics"]["mean_iou"],
            jnpa_dense["soft_unet"]["final_val_metrics"]["mean_iou"],
        ],
        "JNPA dense validation comparison",
    )
    jnpa_class_chart = save_class_accuracy_chart(
        ASSET_DIR / "jnpa_class_chart_v3.png",
        ["Water", "Industrial", "Bare land", "Vegetation", "Urban"],
        jnpa_dense["teacher_baseline_metrics"]["class_accuracy"],
        jnpa_dense["hard_unet"]["final_val_metrics"]["class_accuracy"],
        jnpa_dense["soft_unet"]["final_val_metrics"]["class_accuracy"],
        "JNPA class-wise accuracy",
    )

    cart_overall = save_overall_metrics_chart(
        ASSET_DIR / "cart_overall_chart_v3.png",
        ["Teacher baseline", "Hard U-Net", "Soft U-Net"],
        [
            cart_dense["teacher_baseline_metrics"]["pixel_accuracy"],
            cart_dense["hard_unet"]["final_val_metrics"]["pixel_accuracy"],
            cart_dense["soft_unet"]["final_val_metrics"]["pixel_accuracy"],
        ],
        [
            cart_dense["teacher_baseline_metrics"]["mean_iou"],
            cart_dense["hard_unet"]["final_val_metrics"]["mean_iou"],
            cart_dense["soft_unet"]["final_val_metrics"]["mean_iou"],
        ],
        "CARTOSAT dense validation comparison",
    )
    cart_class_chart = save_class_accuracy_chart(
        ASSET_DIR / "cart_class_chart_v3.png",
        ["Water", "Dense urban", "Port infra", "Terrain"],
        cart_dense["teacher_baseline_metrics"]["class_accuracy"],
        cart_dense["hard_unet"]["final_val_metrics"]["class_accuracy"],
        cart_dense["soft_unet"]["final_val_metrics"]["class_accuracy"],
        "CARTOSAT class-wise accuracy",
    )

    slide_paths = [
        create_cover_slide(
            ASSET_DIR / "slide_01_cover.png",
            jnpa_preview,
            cart_preview,
            ROOT / jnpa_dense["soft_unet"]["prediction_maps"]["overlay_path"],
            ROOT / cart_dense["hard_unet"]["prediction_maps"]["overlay_path"],
            jnpa_dense,
            cart_dense,
        ),
        create_dataset_slide(ASSET_DIR / "slide_02_datasets.png", jnpa_preview, cart_preview),
        create_dataset_detail_slide(
            ASSET_DIR / "slide_03_dataset_detail.png",
            jnpa_preview,
            cart_preview,
            jnpa_dense,
            cart_dense,
        ),
        create_dataset_process_slide(
            ASSET_DIR / "slide_04_jnpa_process.png",
            "JNPA: What We Did Step by Step",
            "From raw JNPA raster to the final promoted soft-label dense U-Net model",
            [
                ("JNPA source raster", "Input scene used for the project", jnpa_preview),
                (
                    "JNPA patch-classifier stage",
                    f"V2 patch classifier | {jnpa_v2['metrics']['val_accuracy'] * 100:.2f}% validation accuracy",
                    ROOT / "outputs/jnpa_patch_classifier_v2/predictions/patch_classifier_v2_overlay.png",
                ),
                (
                    "JNPA final dense output",
                    f"Soft U-Net V128 | {jnpa_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {jnpa_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mIoU",
                    ROOT / jnpa_dense["soft_unet"]["prediction_maps"]["overlay_path"],
                ),
            ],
            [
                (
                    "Step 1 | Source Preparation",
                    "We loaded the JNPA GeoTIFF, preserved its geospatial metadata, and normalized the single panchromatic band to a stable grayscale range for downstream processing.",
                    TEAL,
                    (247, 252, 250),
                ),
                (
                    "Step 2 | Pseudo-label Creation",
                    "The raster was split into 256x256 teacher patches, low-information patches were filtered, and clusters were manually mapped into 5 classes after visual inspection.",
                    GOLD,
                    (255, 250, 241),
                ),
                (
                    "Step 3 | Intermediate Learning",
                    "The first direct U-Net underperformed because the supervision was too patch-level. A patch classifier then validated that the semantic grouping was meaningful.",
                    BLUE,
                    (244, 248, 255),
                ),
                (
                    "Step 4 | Final Dense Upgrade",
                    "A pixel teacher generated hard masks and soft probabilities. Retraining at 128x128 showed that the soft-label U-Net was the best final JNPA model.",
                    GREEN,
                    (245, 252, 246),
                ),
            ],
            "JNPA ultimately favored soft labels because mixed coastal transitions and uncertain boundaries benefited from probability-based supervision.",
        ),
        create_dataset_process_slide(
            ASSET_DIR / "slide_05_cart_process.png",
            "CARTOSAT: What We Did Step by Step",
            "From a simpler patch setup to the final 4-class hard-mask dense model",
            [
                ("CARTOSAT source raster", "Input scene used for the project", cart_preview),
                (
                    "CARTOSAT patch-classifier stage",
                    f"V3 4-class patch classifier | {cart_v3['metrics']['val_accuracy'] * 100:.2f}% validation accuracy",
                    ROOT / "outputs/cartosat_patch_classifier_v3_4class/predictions/patch_classifier_v3_overlay.png",
                ),
                (
                    "CARTOSAT final dense output",
                    f"Hard U-Net V128 | {cart_dense['best_variant_metrics']['pixel_accuracy'] * 100:.2f}% pixel accuracy | {cart_dense['best_variant_metrics']['mean_iou'] * 100:.2f}% mIoU",
                    ROOT / cart_dense["hard_unet"]["prediction_maps"]["overlay_path"],
                ),
            ],
            [
                (
                    "Step 1 | Source Preparation",
                    "We loaded the 1 m CARTOSAT raster, normalized the grayscale range, and preserved the spatial metadata needed for full-scene reconstruction and GeoTIFF export.",
                    TEAL,
                    (247, 252, 250),
                ),
                (
                    "Step 2 | Class Design Refinement",
                    "The early simpler CARTOSAT formulation was expanded so that port and waterfront infrastructure became its own class instead of being merged into generic urban texture.",
                    GOLD,
                    (255, 250, 241),
                ),
                (
                    "Step 3 | Intermediate Patch Modeling",
                    "The 4-class patch classifier established a stronger semantic structure for water, dense urban built-up, port infrastructure, and terrain/open ground.",
                    BLUE,
                    (244, 248, 255),
                ),
                (
                    "Step 4 | Final Dense Upgrade",
                    "Dense pseudo masks were generated and both hard-mask and soft-label U-Nets were trained at 128x128. Hard-mask U-Net performed best and became the production model.",
                    GREEN,
                    (245, 252, 246),
                ),
            ],
            "CARTOSAT favored hard masks because its engineered waterfront and urban structures were sharper and more separable than the mixed coastal transitions seen in JNPA.",
        ),
        create_labeling_slide(
            ASSET_DIR / "slide_06_labeling.png",
            ROOT / jnpa_dense["artifacts"]["dense_preview_panel"],
        ),
        create_evolution_slide(
            ASSET_DIR / "slide_07_evolution.png",
            jnpa_v1,
            jnpa_v2,
            jnpa_dense,
            cart_v3,
            cart_dense,
        ),
        create_pipeline_slide(ASSET_DIR / "slide_08_pipeline.png"),
        create_quant_slide(
            ASSET_DIR / "slide_09_jnpa_quant.png",
            "JNPA Final Results",
            "Dense-label upgrade results after moving to 128x128 student patches",
            jnpa_overall,
            jnpa_class_chart,
            [
                (
                    "Teacher Baseline",
                    f"Pixel accuracy: {jnpa_dense['teacher_baseline_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {jnpa_dense['teacher_baseline_metrics']['mean_iou'] * 100:.2f}%",
                    SLATE,
                    (248, 248, 248),
                ),
                (
                    "Hard U-Net",
                    f"Pixel accuracy: {jnpa_dense['hard_unet']['final_val_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {jnpa_dense['hard_unet']['final_val_metrics']['mean_iou'] * 100:.2f}%",
                    ORANGE,
                    (255, 248, 242),
                ),
                (
                    "Soft U-Net",
                    f"Pixel accuracy: {jnpa_dense['soft_unet']['final_val_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {jnpa_dense['soft_unet']['final_val_metrics']['mean_iou'] * 100:.2f}%",
                    BLUE,
                    (244, 248, 255),
                ),
                (
                    "Selected Model",
                    "Soft labels won for JNPA because they handled mixed coastal transitions and uncertain boundaries better than hard masks.",
                    GOLD,
                    (255, 250, 241),
                ),
            ],
            "Metrics on this slide are against the dense pseudo-label validation setup used inside the repository.",
        ),
        create_map_comparison_slide(
            ASSET_DIR / "slide_10_jnpa_maps.png",
            "JNPA Map Comparison",
            "Teacher baseline versus final promoted soft-label dense model",
            ROOT / jnpa_dense["teacher_baseline_maps"]["overlay_path"],
            ROOT / jnpa_dense["soft_unet"]["prediction_maps"]["overlay_path"],
            "Teacher baseline reconstruction",
            "Final soft-label dense reconstruction",
            (
                jnpa_dense["teacher_baseline_metrics"]["pixel_accuracy"],
                jnpa_dense["teacher_baseline_metrics"]["mean_iou"],
            ),
            (
                jnpa_dense["soft_unet"]["final_val_metrics"]["pixel_accuracy"],
                jnpa_dense["soft_unet"]["final_val_metrics"]["mean_iou"],
            ),
            f"Mean IoU improved by {jnpa_dense['comparison']['mean_iou_improvement'] * 100:.2f} percentage points after the dense-label upgrade.",
            [
                "Water and industrial zones became much more spatially coherent.",
                "The final output no longer forces one class over an entire 256x256 patch.",
                "Urban and vegetation transitions are cleaner because the model sees denser supervision at 128x128 scale.",
            ],
        ),
        create_quant_slide(
            ASSET_DIR / "slide_11_cart_quant.png",
            "CARTOSAT Final Results",
            "Final 4-class dense model comparison with hard-mask and soft-label training",
            cart_overall,
            cart_class_chart,
            [
                (
                    "Teacher Baseline",
                    f"Pixel accuracy: {cart_dense['teacher_baseline_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {cart_dense['teacher_baseline_metrics']['mean_iou'] * 100:.2f}%",
                    SLATE,
                    (248, 248, 248),
                ),
                (
                    "Hard U-Net",
                    f"Pixel accuracy: {cart_dense['hard_unet']['final_val_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {cart_dense['hard_unet']['final_val_metrics']['mean_iou'] * 100:.2f}%",
                    ORANGE,
                    (255, 248, 242),
                ),
                (
                    "Soft U-Net",
                    f"Pixel accuracy: {cart_dense['soft_unet']['final_val_metrics']['pixel_accuracy'] * 100:.2f}%\n"
                    f"Mean IoU: {cart_dense['soft_unet']['final_val_metrics']['mean_iou'] * 100:.2f}%",
                    BLUE,
                    (244, 248, 255),
                ),
                (
                    "Selected Model",
                    "Hard masks won for CARTOSAT because engineered surfaces and waterfront layouts were sharper and more separable.",
                    GOLD,
                    (255, 250, 241),
                ),
            ],
            "CARTOSAT uses the richer 4-class formulation in the final production model, including a separate port/waterfront infrastructure class.",
        ),
        create_map_comparison_slide(
            ASSET_DIR / "slide_12_cart_maps.png",
            "CARTOSAT Map Comparison",
            "Teacher baseline versus final promoted hard-mask dense model",
            ROOT / cart_dense["teacher_baseline_maps"]["overlay_path"],
            ROOT / cart_dense["hard_unet"]["prediction_maps"]["overlay_path"],
            "Teacher baseline reconstruction",
            "Final hard-mask dense reconstruction",
            (
                cart_dense["teacher_baseline_metrics"]["pixel_accuracy"],
                cart_dense["teacher_baseline_metrics"]["mean_iou"],
            ),
            (
                cart_dense["hard_unet"]["final_val_metrics"]["pixel_accuracy"],
                cart_dense["hard_unet"]["final_val_metrics"]["mean_iou"],
            ),
            f"Mean IoU improved by {cart_dense['comparison']['mean_iou_improvement'] * 100:.2f} percentage points after dense-label training.",
            [
                "Water stays dominant and coherent across the harbor scene.",
                "Port and waterfront infrastructure become a meaningful separate class instead of blending into generic urban texture.",
                "The final map supports more useful land-use interpretation for the deployed platform.",
            ],
        ),
        create_platform_slide(
            ASSET_DIR / "slide_13_platform.png",
            JNPA_PLATFORM_RESULT,
            CARTOSAT_PLATFORM_RESULT,
        ),
        create_conclusion_slide(ASSET_DIR / "slide_14_conclusion.png", jnpa_dense, cart_dense),
    ]

    deck_base = "Urban_Feature_Extraction_Final_Presentation_2026_03_25_v5"
    key_path = PRESENTATION_DIR / f"{deck_base}.key"
    pptx_path = PRESENTATION_DIR / f"{deck_base}.pptx"

    script_lines = [
        'tell application "Keynote"',
        "activate",
        'set docRef to make new document with properties {document theme:theme "White"}',
        'set blankMaster to master slide "Blank" of docRef',
        'tell slide 1 of docRef',
        'set object text of default title item to ""',
        'set object text of default body item to ""',
        f'make new image with properties {{file:(POSIX file "{slide_paths[0]}"), position:{{0, 0}}, width:{SLIDE_W}, height:{SLIDE_H}}}',
        "end tell",
    ]

    for slide_path in slide_paths[1:]:
        script_lines.extend(
            [
                "",
                'set newSlide to make new slide at end of slides of docRef with properties {base slide:blankMaster}',
                "tell newSlide",
                f'make new image with properties {{file:(POSIX file "{slide_path}"), position:{{0, 0}}, width:{SLIDE_W}, height:{SLIDE_H}}}',
                "end tell",
            ]
        )

    script_lines.extend(
        [
            "",
            f'save docRef in POSIX file "{key_path}"',
            f'export docRef to POSIX file "{pptx_path}" as Microsoft PowerPoint',
            "close docRef saving no",
            "end tell",
        ]
    )

    try:
        subprocess.run(["osascript"], input="\n".join(script_lines).encode("utf-8"), check=True)
        if FORCE_WIDESCREEN_PPTX:
            build_pptx_from_template(slide_paths, pptx_path)
        return key_path, pptx_path
    except subprocess.CalledProcessError:
        build_pptx_from_template(slide_paths, pptx_path)
        return None, pptx_path


if __name__ == "__main__":
    key_path, pptx_path = build_presentation()
    print(
        json.dumps(
            {
                "keynote": str(key_path) if key_path else None,
                "pptx": str(pptx_path),
            },
            indent=2,
        )
    )
