"""Shared dashboard toolkit: inline SVG icons + small HTML helpers.

Every dashboard in this repository imports from here so that:
  * there is exactly ONE place icons are defined (and they are SVG, never emoji),
  * the visual language is consistent across all six projects.

Copy this file into a project (or import it if the project is on the path).
There is no external dependency and no CDN.
"""
from __future__ import annotations

import html

# --------------------------------------------------------------------- icons --
# 24x24 stroke icons, currentColor so they inherit the surrounding text colour.
ICONS = {
    "dashboard": '<path d="M3 3h8v8H3zM13 3h8v5h-8zM13 12h8v9h-8zM3 15h8v6H3z"/>',
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/>'
                '<path d="M3 10h18M8 3v4M16 3v4"/>',
    "camera": '<rect x="3" y="7" width="18" height="13" rx="2"/>'
              '<circle cx="12" cy="13" r="4"/><path d="M8 7l1-3h6l1 3"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "chart": '<path d="M4 20V6M4 20h16"/><path d="M8 17v-6M12 17V9M16 17v-4"/>',
    "trend": '<path d="M3 17l6-6 4 4 7-7"/><path d="M14 8h6v6"/>',
    "signal": '<path d="M4 20v-4M9 20v-9M14 20v-14M19 20V3"/>',
    "database": '<ellipse cx="12" cy="6" rx="8" ry="3"/>'
                '<path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 '
                '1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chat": '<path d="M4 5h16v11H8l-4 4z"/><path d="M8 9h8M8 12h5"/>',
    "brain": '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5 3 3 0 0 0 2 4 3 3 0 '
             '0 0 5 1V4a3 3 0 0 0-3 0z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 '
             '1 1 5 3 3 0 0 1-2 4 3 3 0 0 1-5 1V4a3 3 0 0 1 3 0z"/>',
    "heart": '<path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2 4 4 0 0 1 7 2c0 '
             '5.5-7 10-7 10z"/>',
    "scan": '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 '
            '1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M4 12h16"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 '
            '2-2V8z"/><path d="M14 3v5h5"/>',
    "warning": '<path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/>',
    "check": '<path d="M4 12l5 5L20 6"/>',
    "map": '<path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2-6-2z"/><path d="M9 4v14M15 '
           '6v14"/>',
    "robot": '<rect x="5" y="8" width="14" height="10" rx="2"/>'
             '<path d="M12 4v4M8 21v-3M16 21v-3"/><circle cx="9" cy="13" '
             'r="1"/><circle cx="15" cy="13" r="1"/>',
    "route": '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/>'
             '<path d="M6 8v6a4 4 0 0 0 4 4h6"/>',
    "device": '<rect x="7" y="3" width="10" height="18" rx="2"/><path '
              'd="M11 18h2"/>',
    "leaf": '<path d="M5 19c0-8 6-14 14-14 0 8-6 14-14 14z"/><path d="M5 19l7-7"/>',
    "droplet": '<path d="M12 3s6 6 6 10a6 6 0 0 1-12 0c0-4 6-10 6-10z"/>',
    "thermometer": '<path d="M14 14V5a2 2 0 1 0-4 0v9a4 4 0 1 0 4 0z"/>',
    "rain": '<path d="M6 15a4 4 0 0 1 0-8 5 5 0 0 1 9-2 4 4 0 0 1 3 10"/>'
            '<path d="M8 18l-1 3M12 18l-1 3M16 18l-1 3"/>',
    "gauge": '<path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="M12 '
             '12l4-4"/><path d="M3 12a9 9 0 1 1 18 0"/>',
    "pump": '<circle cx="9" cy="9" r="3"/><path d="M12 9h6a2 2 0 0 1 2 2v6M3 '
            '20h16"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 '
            '18h.01"/>',
    "power": '<path d="M12 3v9"/><path d="M6.5 6.5a8 8 0 1 0 11 0"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
}


def icon(name: str, size: int = 18) -> str:
    """Return an inline SVG for `name` (empty string if unknown)."""
    path = ICONS.get(name)
    if not path:
        return ""
    return (f'<svg class="ico" viewBox="0 0 24 24" width="{size}" height="{size}"'
            f' fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


# ---------------------------------------------------------------- components --
def card(title: str, icon_name: str, body: str, span: int = 1,
         note: str = "") -> str:
    note_html = f'<p class="src">{html.escape(note)}</p>' if note else ""
    return (f'<section class="card" style="grid-column:span {span}">'
            f'<h2>{icon(icon_name)}<span>{html.escape(title)}</span></h2>'
            f'{body}{note_html}</section>')


def stat(label: str, value, unit: str = "", icon_name: str = "gauge") -> str:
    return (f'<div class="stat"><div class="stat-ico">{icon(icon_name)}</div>'
            f'<div><div class="stat-val">{html.escape(str(value))}'
            f'<small>{html.escape(unit)}</small></div>'
            f'<div class="stat-lab">{html.escape(label)}</div></div></div>')


def badge(text: str, tone: str = "") -> str:
    return f'<span class="badge {tone}">{html.escape(text)}</span>'


def table(headers, rows) -> str:
    if not rows:
        return '<p class="muted">No rows.</p>'
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(c)) if c not in (None, '') else '&mdash;'}</td>"
            for c in r) + "</tr>" for r in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def sparkline(values, color: str = "#3b82f6", height: int = 56,
              width: int = 240) -> str:
    pts = [v for v in values if v is not None]
    if len(pts) < 2:
        return '<p class="muted">Not enough data to plot.</p>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    step = width / (len(pts) - 1)
    coords = " ".join(
        f"{i * step:.1f},{height - (v - lo) / rng * (height - 6) - 3:.1f}"
        for i, v in enumerate(pts))
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none"><polyline fill="none" stroke="{color}" '
            f'stroke-width="2" points="{coords}"/></svg>')


def bar_chart(labels, values, color: str = "#3b82f6", height: int = 140) -> str:
    if not values:
        return '<p class="muted">No data.</p>'
    hi = max(values) or 1.0
    n = len(values)
    w = 100 / max(n, 1)
    bars = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = (v / hi) * 100
        bars.append(f'<div class="bar" style="left:{i * w:.3f}%;'
                    f'width:{w:.3f}%" title="{html.escape(str(lab))}: {v}">'
                    f'<span style="height:{h:.1f}%;background:{color}"></span></div>')
    return f'<div class="barchart" style="height:{height}px">' + "".join(bars) + "</div>"


BASE_CSS = """
 :root{--bg:#0f1115;--panel:#171a21;--line:#242833;--txt:#e8e8e8;--muted:#8b93a7;
       --accent:#3b82f6;--ok:#22c55e;--warn:#f59e0b;--crit:#ef4444}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--txt);
      font:14px/1.5 system-ui,Segoe UI,Roboto,Arial,sans-serif}
 header{display:flex;align-items:center;gap:12px;padding:14px 20px;
        background:var(--panel);border-bottom:1px solid var(--line)}
 header .brand{color:var(--accent)}
 header h1{font-size:16px;margin:0;font-weight:600}
 header .sub{color:var(--muted);font-size:12px}
 header nav{margin-left:auto;display:flex;gap:14px}
 header nav a{color:var(--muted);text-decoration:none;font-size:12px}
 header nav a:hover{color:var(--txt)}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
       gap:16px;padding:20px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
       padding:16px}
 .card h2{display:flex;align-items:center;gap:8px;margin:0 0 12px;font-size:12px;
          text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
          font-weight:600}
 .card h2 svg{color:var(--accent)}
 .stat{display:flex;align-items:center;gap:12px;margin:10px 0}
 .stat-ico{color:var(--accent)}
 .stat-val{font-size:22px;font-weight:700}
 .stat-val small{font-size:12px;color:var(--muted);margin-left:4px}
 .stat-lab{font-size:12px;color:var(--muted)}
 .src{font-size:11px;color:var(--muted);margin:10px 0 0;
      border-top:1px dashed var(--line);padding-top:8px}
 .muted{color:var(--muted);font-size:12px}
 .spark{width:100%;height:56px;display:block}
 .barchart{position:relative;display:flex;align-items:flex-end;
           border-bottom:1px solid var(--line)}
 .bar{position:absolute;bottom:0;height:100%;display:flex;align-items:flex-end;
      padding:0 1px}
 .bar span{width:100%;border-radius:2px 2px 0 0;min-height:1px;display:block}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
    letter-spacing:.05em}
 .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
        border:1px solid var(--line);color:var(--muted)}
 .badge.ok{border-color:var(--ok);color:var(--ok)}
 .badge.warn{border-color:var(--warn);color:var(--warn)}
 .badge.crit{border-color:var(--crit);color:var(--crit)}
 footer{color:var(--muted);font-size:12px;padding:0 20px 24px}
"""


def page(title: str, subtitle: str, cards: str, footer: str = "",
         nav=("people", "attendance", "api")) -> str:
    nav_html = "".join(f'<a href="#{n}">{html.escape(n)}</a>' for n in nav)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{BASE_CSS}</style></head><body>"
        f'<header><span class="brand">{icon("dashboard", 20)}</span>'
        f"<h1>{html.escape(title)}</h1>"
        f'<span class="sub">{html.escape(subtitle)}</span>'
        f"<nav>{nav_html}</nav></header>"
        f'<div class="grid">{cards}</div>'
        + (f"<footer>{html.escape(footer)}</footer>" if footer else "")
        + "</body></html>"
    )
