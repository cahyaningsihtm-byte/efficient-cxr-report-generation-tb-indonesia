#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

KD_CSV = ROOT / "medalign_lite" / "outputs" / "eval_kd_qwenfp16-chex_l4_fold0" / (
    "chexagent_kd_qwenfp16_reprkd_fold0_native_prompted_symmetric_v3"
) / "tb14_per_class_symmetric.csv"
STUDENT_NATIVE_CSV = ROOT / "medalign_lite" / "outputs" / "eval_baseline_chexagent_l4_fold0" / (
    "chexagent_baseline_fold0_symmetric_v3"
) / "tb14_per_class_symmetric.csv"
STUDENT_PROMPTED_CSV = ROOT / "medalign_lite" / "outputs" / "eval_baseline_chexagent_l4_fold0" / (
    "chexagent_baseline_fold0_prompted_canonical_clean_symmetric_v3"
) / "tb14_per_class_chexagent_prompt_L4.csv"

OUT_DIR = ROOT / "outputs" / "paper_figures"
OUT_PNG = OUT_DIR / "tb14_delta_f1_report_reprkd_vs_student.png"
OUT_SVG = OUT_DIR / "tb14_delta_f1_report_reprkd_vs_student.svg"
OUT_CSV = OUT_DIR / "tb14_delta_f1_report_reprkd_vs_student.csv"

LABEL_DISPLAY = {
    "apical_changes": "Apical changes",
    "atelectasis": "Atelectasis",
    "bronchial_changes": "Bronchial changes",
    "bulla": "Bulla",
    "cardiomegaly": "Cardiomegaly",
    "cavity": "Cavity",
    "consolidation": "Consolidation",
    "fibrosis": "Fibrosis",
    "hilar_enlargement": "Hilar enlargement",
    "infiltration": "Infiltration",
    "mediastinum_displaced": "Mediastinal displacement",
    "pleural_abnormality": "Pleural abnormality",
    "pleural_effusion": "Pleural effusion",
    "pneumothorax": "Pneumothorax",
}

CORE_TB_PATTERN_ORDER = [
    "apical_changes",
    "cavity",
    "consolidation",
    "infiltration",
    "hilar_enlargement",
    "pleural_effusion",
    "pleural_abnormality",
]

SECONDARY_LABEL_ORDER = [
    "fibrosis",
    "bronchial_changes",
    "atelectasis",
    "pneumothorax",
    "mediastinum_displaced",
    "cardiomegaly",
    "bulla",
]

CORE_TB_PATTERN_LABELS = set(CORE_TB_PATTERN_ORDER)
LABEL_ORDER = {
    label: idx for idx, label in enumerate(CORE_TB_PATTERN_ORDER + SECONDARY_LABEL_ORDER)
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def labelize(s: str) -> str:
    return LABEL_DISPLAY.get(s, " ".join(part.capitalize() for part in s.split("_")))


def as_float(v: str | float | int | None) -> float:
    try:
        return float(v or 0)
    except ValueError:
        return 0.0


def by_label_and_prompt(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in rows:
        out[(row["tb_label"], row["prompt_mode"])] = row
    return out


def read_metric_table() -> list[dict[str, float | str]]:
    kd = by_label_and_prompt(read_rows(KD_CSV))
    native_student = by_label_and_prompt(read_rows(STUDENT_NATIVE_CSV))
    prompted_student = by_label_and_prompt(read_rows(STUDENT_PROMPTED_CSV))

    labels = sorted({key[0] for key in kd})
    rows = []
    for label in labels:
        kd_native = kd.get((label, "native"), {})
        kd_prompted = kd.get((label, "prompted"), {})
        st_native = native_student.get((label, "native"), {})
        st_prompted = prompted_student.get((label, "prompted"), {})

        native_delta = as_float(kd_native.get("f1")) - as_float(st_native.get("f1"))
        prompted_delta = as_float(kd_prompted.get("f1")) - as_float(st_prompted.get("f1"))
        support = int(as_float(kd_native.get("support") or kd_prompted.get("support")))

        rows.append(
            {
                "abnormality": labelize(label),
                "tb_label": label,
                "support": support,
                "student_native_f1": as_float(st_native.get("f1")),
                "student_prompted_f1": as_float(st_prompted.get("f1")),
                "kd_native_f1": as_float(kd_native.get("f1")),
                "kd_prompted_f1": as_float(kd_prompted.get("f1")),
                "delta_native": native_delta,
                "delta_prompted": prompted_delta,
                "sort_key": max(abs(native_delta), abs(prompted_delta)),
                "is_core_tb_pattern": label in CORE_TB_PATTERN_LABELS,
            }
        )

    rows.sort(key=lambda r: LABEL_ORDER.get(str(r["tb_label"]), 999))
    return rows


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def pp(value: float) -> str:
    return f"{value * 100:+.1f}"


def draw_chart(rows: list[dict[str, float | str]], png_path: Path) -> None:
    scale = 2
    width, height = 2600, 1800
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    subtitle_f = font(42, bold=True)
    group_f = font(27, bold=True)
    axis_f = font(28)
    label_f = font(34)
    label_bold_f = font(34, bold=True)
    value_f = font(32, bold=True)
    legend_f = font(34)
    note_f = font(24)

    ink = "#111827"
    muted = "#5B6472"
    rare_muted = "#8A94A6"
    grid = "#D7DEE8"
    native_color = "#0B6E8A"
    prompted_color = "#E87532"
    pos_bg = "#EAF6EF"
    neg_bg = "#FBEEF0"

    left_label_x = 80
    plot_left = 750
    plot_right = 2400
    zero_x = (plot_left + plot_right) // 2
    top = 275
    row_h = 82
    group_gap = 46
    bar_h = 24
    core_count = sum(1 for row in rows if row["is_core_tb_pattern"])
    plot_body_h = row_h * len(rows) + group_gap
    max_abs = 0.45
    ticks = [-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]

    draw.text(
        (80, 55),
        "Per-abnormality F1 change after CheX-QF16-RRKD",
        fill=ink,
        font=subtitle_f,
    )

    legend_y = 122
    draw.rounded_rectangle((80, legend_y + 7, 116, legend_y + 36), radius=6, fill=native_color)
    draw.text((128, legend_y), "RRKD native vs. CheXAgent native", fill=ink, font=legend_f)
    draw.rounded_rectangle((690, legend_y + 7, 726, legend_y + 36), radius=6, fill=prompted_color)
    draw.text((738, legend_y), "RRKD prompted vs. CheXAgent prompted", fill=ink, font=legend_f)

    draw.text((left_label_x, top - 48), "Clinically central TB-pattern categories", fill=ink, font=group_f)
    secondary_y = top + core_count * row_h + group_gap
    draw.text((left_label_x, secondary_y - 48), "Associated / secondary / completeness categories", fill=rare_muted, font=group_f)

    for tick in ticks:
        x = zero_x + int((tick / max_abs) * ((plot_right - plot_left) / 2))
        draw.line((x, top - 24, x, top + plot_body_h + 18), fill=grid, width=2)
        label = "0" if tick == 0 else f"{tick * 100:+.0f}"
        tw = text_width(draw, label, axis_f)
        draw.text((x - tw / 2, top + plot_body_h + 30), label, fill=muted, font=axis_f)
    draw.line((zero_x, top - 35, zero_x, top + plot_body_h + 18), fill="#475569", width=4)
    axis_label = "\u0394 F1 (percentage points)"
    tw = text_width(draw, axis_label, axis_f)
    draw.text(((plot_left + plot_right - tw) / 2, top + plot_body_h + 72), axis_label, fill=muted, font=axis_f)

    draw.rounded_rectangle((plot_left, top - 35, zero_x, top + plot_body_h + 18), radius=16, fill=neg_bg)
    draw.rounded_rectangle((zero_x, top - 35, plot_right, top + plot_body_h + 18), radius=16, fill=pos_bg)
    for tick in ticks:
        x = zero_x + int((tick / max_abs) * ((plot_right - plot_left) / 2))
        draw.line((x, top - 24, x, top + plot_body_h + 18), fill=grid if tick else "#475569", width=2 if tick else 4)

    for idx, row in enumerate(rows):
        is_core = bool(row["is_core_tb_pattern"])
        y = top + idx * row_h + (group_gap if idx >= core_count else 0)
        name = str(row["abnormality"])
        support = int(row["support"])
        row_ink = ink if is_core else rare_muted
        label_font = label_bold_f if is_core else label_f
        draw.text((left_label_x, y + 16), name, fill=row_ink, font=label_font)
        rare_mark = "*" if support < 20 else ""
        draw.text((540, y + 18), f"n={support}{rare_mark}", fill=rare_muted if support < 20 else muted, font=note_f)

        if idx % 2:
            draw.line((left_label_x, y + row_h - 2, plot_right, y + row_h - 2), fill="#EEF2F7", width=2)

        for lane, key, color in [
            (-1, "delta_native", native_color),
            (1, "delta_prompted", prompted_color),
        ]:
            val = float(row[key])
            x = zero_x + int((val / max_abs) * ((plot_right - plot_left) / 2))
            y_bar = y + 20 + (lane + 1) * 15
            x0, x1 = sorted([zero_x, x])
            if abs(x1 - x0) < 3:
                x1 = x0 + 3
            draw.rounded_rectangle((x0, y_bar, x1, y_bar + bar_h), radius=7, fill=color)

            label = pp(val)
            tw = text_width(draw, label, value_f)
            if val >= 0:
                lx = min(x1 + 10, plot_right - tw)
            else:
                lx = max(x0 - tw - 10, plot_left)
            draw.text((lx, y_bar - 2), label, fill=color, font=value_f)

    note = (
        "Positive values indicate RRKD improved per-class F1 over the matched CheXAgent baseline. "
        "n indicates reference-positive cases per label."
    )
    draw.text((80, height - 115), note, fill=muted, font=note_f)
    source = "Labels with n<20 should be interpreted cautiously. Source: TB-14 symmetric per-class evaluation; RRKD, Qwen-FP16 Prompted Teacher, Testing Fold."
    draw.text((80, height - 72), source, fill=muted, font=note_f)

    img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    img.save(png_path, dpi=(300, 300))


def svg_text(x: float, y: float, text: str, size: int, weight: str = "400", fill: str = "#111827", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def write_svg(rows: list[dict[str, float | str]], svg_path: Path) -> None:
    width, height = 2600, 1800
    plot_left, plot_right = 750, 2400
    zero_x = (plot_left + plot_right) / 2
    top, row_h, bar_h = 275, 82, 24
    group_gap = 46
    core_count = sum(1 for row in rows if row["is_core_tb_pattern"])
    plot_body_h = row_h * len(rows) + group_gap
    max_abs = 0.45
    ticks = [-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]
    native_color = "#0B6E8A"
    prompted_color = "#E87532"
    tb_marker = "#2B2F38"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(80, 90, "Per-abnormality F1 change after CheX-QF16-RRKD", 42, "700"),
        f'<rect x="80" y="129" width="36" height="29" rx="6" fill="{native_color}"/>',
        svg_text(128, 156, "RRKD native vs. CheXAgent native", 34),
        f'<rect x="690" y="129" width="36" height="29" rx="6" fill="{prompted_color}"/>',
        svg_text(738, 156, "RRKD prompted vs. CheXAgent prompted", 34),
        svg_text(80, top - 25, "Clinically central TB-pattern categories", 27, "700"),
        svg_text(80, top + core_count * row_h + group_gap - 25, "Associated / secondary / completeness categories", 27, "700", "#8A94A6"),
        f'<rect x="{plot_left}" y="{top - 35}" width="{zero_x - plot_left}" height="{plot_body_h + 53}" rx="16" fill="#FBEEF0"/>',
        f'<rect x="{zero_x}" y="{top - 35}" width="{plot_right - zero_x}" height="{plot_body_h + 53}" rx="16" fill="#EAF6EF"/>',
    ]

    for tick in ticks:
        x = zero_x + (tick / max_abs) * ((plot_right - plot_left) / 2)
        color = "#475569" if tick == 0 else "#D7DEE8"
        width_line = 4 if tick == 0 else 2
        label = "0" if tick == 0 else f"{tick * 100:+.0f}"
        parts.append(f'<line x1="{x:.1f}" y1="{top - 24}" x2="{x:.1f}" y2="{top + plot_body_h + 18}" stroke="{color}" stroke-width="{width_line}"/>')
        parts.append(svg_text(x, top + plot_body_h + 58, label, 28, "400", "#5B6472", "middle"))

    for idx, row in enumerate(rows):
        is_core = bool(row["is_core_tb_pattern"])
        y = top + idx * row_h + (group_gap if idx >= core_count else 0)
        support = int(row["support"])
        row_fill = "#111827" if is_core else "#8A94A6"
        row_weight = "700" if is_core else "400"
        parts.append(svg_text(80, y + 45, str(row["abnormality"]), 34, row_weight, row_fill))
        rare_mark = "*" if support < 20 else ""
        parts.append(svg_text(540, y + 45, f'n={support}{rare_mark}', 24, "400", "#8A94A6" if support < 20 else "#5B6472"))
        if idx % 2:
            parts.append(f'<line x1="80" y1="{y + row_h - 2}" x2="{plot_right}" y2="{y + row_h - 2}" stroke="#EEF2F7" stroke-width="2"/>')
        for lane, key, color in [(-1, "delta_native", native_color), (1, "delta_prompted", prompted_color)]:
            val = float(row[key])
            x = zero_x + (val / max_abs) * ((plot_right - plot_left) / 2)
            x0, x1 = sorted([zero_x, x])
            if abs(x1 - x0) < 3:
                x1 = x0 + 3
            y_bar = y + 20 + (lane + 1) * 15
            parts.append(f'<rect x="{x0:.1f}" y="{y_bar}" width="{x1 - x0:.1f}" height="{bar_h}" rx="7" fill="{color}"/>')
            label = pp(val)
            lx = min(x1 + 12, plot_right - 90) if val >= 0 else max(x0 - 12, plot_left + 90)
            anchor = "start" if val >= 0 else "end"
            parts.append(svg_text(lx, y_bar + 26, label, 32, "700", color, anchor))

    parts.append(svg_text((plot_left + plot_right) / 2, top + plot_body_h + 105, "\u0394 F1 (percentage points)", 28, "400", "#5B6472", "middle"))
    parts.append(svg_text(80, height - 90, "Positive values indicate RRKD improved per-class F1 over the matched CheXAgent baseline. n indicates reference-positive cases per label.", 24, "400", "#5B6472"))
    parts.append(svg_text(80, height - 50, "Labels with n<20 should be interpreted cautiously. Source: TB-14 symmetric per-class evaluation; RRKD, Qwen-FP16 Prompted Teacher, Testing Fold.", 24, "400", "#5B6472"))
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    cols = [
        "tb_label",
        "abnormality",
        "support",
        "student_native_f1",
        "kd_native_f1",
        "delta_native",
        "student_prompted_f1",
        "kd_prompted_f1",
        "delta_prompted",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in cols})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_metric_table()
    write_delta_csv(rows, OUT_CSV)
    draw_chart(rows, OUT_PNG)
    write_svg(rows, OUT_SVG)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved SVG: {OUT_SVG}")
    print(f"Saved data: {OUT_CSV}")


if __name__ == "__main__":
    main()
