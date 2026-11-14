"""Render the paper's editable dependency-graph YAML as a standalone SVG.

The input uses JSON syntax, which is valid YAML 1.2. Keeping the parser in the
standard library makes the renderer usable without adding a project dependency.
"""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path


COLORS = {
    "definition": ("#dbeafe", "#2563eb"),
    "property": ("#fef3c7", "#d97706"),
    "proposition": ("#dcfce7", "#16a34a"),
    "example": ("#f3e8ff", "#9333ea"),
    "algorithm": ("#e0e7ff", "#4f46e5"),
    "missing": ("#fee2e2", "#dc2626"),
}


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def render(data: dict, output: Path) -> None:
    stages = data["stages"]
    nodes = data["nodes"]
    edges = data["edges"]
    by_stage = {stage["id"]: [] for stage in stages}
    for node in nodes:
        by_stage[node["stage"]].append(node)

    margin_x, top = 55, 145
    node_w, node_h, col_gap = 330, 112, 72
    stage_w = node_w + col_gap
    max_nodes = max(len(items) for items in by_stage.values())
    usable_h = max(1420, max_nodes * 132)
    width = margin_x * 2 + len(stages) * node_w + (len(stages) - 1) * col_gap
    height = top + usable_h + 125

    positions: dict[str, tuple[float, float]] = {}
    for col, stage in enumerate(stages):
        items = by_stage[stage["id"]]
        step = usable_h / max(1, len(items))
        for row, node in enumerate(items):
            x = margin_x + col * stage_w
            y = top + row * step + (step - node_h) / 2
            positions[node["id"]] = (x, y)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.12"/>',
        "</filter>",
        "<style>",
        ".edge{fill:none;stroke:#64748b;stroke-width:1.35;opacity:.28;"
        "marker-end:url(#arrow)} .edge:hover{opacity:1;stroke-width:2.5}",
        ".edge.context{stroke-dasharray:6 4;opacity:.18}",
        ".node-title{font:600 14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "fill:#0f172a} .node-detail{font:12px -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;fill:#334155}",
        ".node-meta{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#64748b}",
        ".stage{font:700 17px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "fill:#0f172a} .heading{font:700 25px -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;fill:#0f172a}",
        "</style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin_x}" y="38" class="heading">{html.escape(data["title"])}</text>',
        f'<text x="{margin_x}" y="64" class="node-detail">'
        "Solid: prerequisite → dependent result. Dashed: context or consequence. "
        "Hover over an edge for its relation.</text>",
    ]

    legend_x = margin_x
    for kind in ("definition", "property", "proposition", "example", "algorithm", "missing"):
        fill, stroke = COLORS[kind]
        out.append(
            f'<rect x="{legend_x}" y="82" width="15" height="15" rx="3" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        out.append(
            f'<text x="{legend_x + 21}" y="94" class="node-detail">{kind}</text>'
        )
        legend_x += 124
    out.append(
        f'<rect x="{legend_x + 5}" y="82" width="15" height="15" rx="3" '
        'fill="#fff" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3"/>'
    )
    out.append(
        f'<text x="{legend_x + 26}" y="94" class="node-detail">'
        "source cleanup needed</text>"
    )

    for col, stage in enumerate(stages):
        x = margin_x + col * stage_w
        out.append(
            f'<text x="{x + node_w / 2}" y="126" text-anchor="middle" '
            f'class="stage">{html.escape(stage["title"])}</text>'
        )

    node_index = {node["id"]: node for node in nodes}
    for edge in edges:
        source = node_index[edge["from"]]
        target = node_index[edge["to"]]
        x1, y1 = positions[source["id"]]
        x2, y2 = positions[target["id"]]
        if source["stage"] < target["stage"]:
            start_x, start_y = x1 + node_w, y1 + node_h / 2
            end_x, end_y = x2, y2 + node_h / 2
            bend = max(35, (end_x - start_x) * 0.42)
            path = (
                f"M {start_x:.1f} {start_y:.1f} "
                f"C {start_x + bend:.1f} {start_y:.1f}, "
                f"{end_x - bend:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}"
            )
        else:
            start_x, start_y = x1 + node_w * 0.72, y1 + node_h
            end_x, end_y = x2 + node_w * 0.28, y2
            offset = 30 + abs(end_y - start_y) * 0.12
            path = (
                f"M {start_x:.1f} {start_y:.1f} "
                f"C {start_x + offset:.1f} {start_y + offset:.1f}, "
                f"{end_x - offset:.1f} {end_y - offset:.1f}, "
                f"{end_x:.1f} {end_y:.1f}"
            )
        kind = edge.get("kind", "dependency")
        title = (
            f'{kind}: {source["title"]} —[{edge["relation"]}]→ {target["title"]}'
        )
        out.append(
            f'<path class="edge {html.escape(kind)}" d="{path}">'
            f'<title>{html.escape(title)}</title></path>'
        )

    for node in nodes:
        x, y = positions[node["id"]]
        fill, stroke = COLORS[node["kind"]]
        warning = bool(node.get("status"))
        border = "#dc2626" if warning else stroke
        dash = ' stroke-dasharray="7 4"' if warning else ""
        out.append(f'<g id="{html.escape(node["id"])}">')
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w}" height="{node_h}" '
            f'rx="10" fill="{fill}" stroke="{border}" stroke-width="'
            f'{"2.2" if warning else "1.4"}"{dash} filter="url(#shadow)"/>'
        )
        title_lines = wrap(node["title"], 39)[:2]
        text_y = y + 21
        for line in title_lines:
            out.append(
                f'<text x="{x + 12:.1f}" y="{text_y:.1f}" class="node-title">'
                f"{html.escape(line)}</text>"
            )
            text_y += 17
        detail_lines = wrap(node["detail"], 49)[:2]
        text_y = max(text_y + 3, y + 57)
        for line in detail_lines:
            out.append(
                f'<text x="{x + 12:.1f}" y="{text_y:.1f}" class="node-detail">'
                f"{html.escape(line)}</text>"
            )
            text_y += 15
        meta = node.get("label", node.get("status", "unlabeled"))
        out.append(
            f'<text x="{x + 12:.1f}" y="{y + node_h - 10:.1f}" class="node-meta">'
            f"{html.escape(meta)} · {html.escape(node['source'])}</text>"
        )
        tooltip = f'{node["kind"]}: {node["title"]}\n{node["detail"]}'
        if node.get("proof_source"):
            tooltip += f'\nproof: {node["proof_source"]}'
        if node.get("status"):
            tooltip += f'\nstatus: {node["status"]}'
        out.append(f"<title>{html.escape(tooltip)}</title></g>")

    out.append(
        f'<text x="{margin_x}" y="{height - 38}" class="node-detail">'
        f'{len(nodes)} nodes · {len(edges)} edges (solid dependencies, dashed context) · '
        "red dashed borders identify manuscript cleanup items</text>"
    )
    out.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("paper_dependency_graph.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/paper_dependency_graph.svg")
    )
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    render(data, args.output)


if __name__ == "__main__":
    main()
