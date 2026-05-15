import argparse
import csv
import json
import math
import site
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GPS CSV (lat/lon/alt) as HTML.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: same name with _gps.html)",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG summary plot",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=None,
        help="Output PNG path (default: same name with _gps.png)",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"doc", "lat_deg", "lon_deg", "alt_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for r in reader:
            rows.append(
                {
                    "doc": int(float(r["doc"])),
                    "lat": float(r["lat_deg"]),
                    "lon": float(r["lon_deg"]),
                    "alt": float(r["alt_m"]),
                }
            )
    if not rows:
        raise ValueError("CSV has no data rows.")
    return rows


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def add_motion_fields(rows):
    if not rows:
        return rows
    rows[0]["step_m"] = 0.0
    rows[0]["cum_m"] = 0.0
    cumulative = 0.0
    for i in range(1, len(rows)):
        a = rows[i - 1]
        b = rows[i]
        step = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        cumulative += step
        b["step_m"] = step
        b["cum_m"] = cumulative
    return rows


def _import_matplotlib_agg():
    user_site = site.getusersitepackages()
    user_paths = [user_site] if isinstance(user_site, str) else list(user_site)
    user_paths = {p for p in user_paths if p}
    if user_paths:
        sys.path = [p for p in sys.path if p not in user_paths]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_png(rows, png_path: Path, title: str):
    plt = _import_matplotlib_agg()
    docs = [r["doc"] for r in rows]
    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    alts = [r["alt"] for r in rows]
    steps = [r["step_m"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_traj, ax_alt, ax_latlon, ax_step = axes.ravel()

    ax_traj.plot(lons, lats, color="#1f77b4", linewidth=1.4)
    ax_traj.scatter([lons[0]], [lats[0]], color="#2ca02c", s=36, label="start", zorder=3)
    ax_traj.scatter([lons[-1]], [lats[-1]], color="#d62728", s=36, label="end", zorder=3)
    ax_traj.set_title("Trajectory (Longitude vs Latitude)")
    ax_traj.set_xlabel("Longitude")
    ax_traj.set_ylabel("Latitude")
    ax_traj.grid(alpha=0.3)
    ax_traj.axis("equal")
    ax_traj.legend(loc="best")

    ax_alt.plot(docs, alts, color="#ff7f0e", linewidth=1.2)
    ax_alt.set_title("Altitude over doc")
    ax_alt.set_xlabel("doc")
    ax_alt.set_ylabel("alt_m")
    ax_alt.grid(alpha=0.3)

    ax_latlon.plot(docs, lats, color="#2ca02c", linewidth=1.2, label="lat_deg")
    ax_latlon.plot(docs, lons, color="#d62728", linewidth=1.2, label="lon_deg")
    ax_latlon.set_title("Lat/Lon over doc")
    ax_latlon.set_xlabel("doc")
    ax_latlon.set_ylabel("degree")
    ax_latlon.grid(alpha=0.3)
    ax_latlon.legend(loc="best")

    ax_step.plot(docs, steps, color="#9467bd", linewidth=1.2)
    ax_step.set_title("Speed Proxy (Step Distance) over doc")
    ax_step.set_xlabel("doc")
    ax_step.set_ylabel("meter/step")
    ax_step.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def build_html(title: str, rows):
    data_json = json.dumps(rows, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Noto Sans KR", sans-serif;
      background: #f4f7fb;
      color: #122230;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 24px auto;
      padding: 0 16px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(11, 27, 54, 0.12);
      padding: 12px;
    }}
    .card h2 {{
      margin: 4px 0 10px;
      font-size: 15px;
      font-weight: 700;
    }}
    svg {{
      width: 100%;
      height: 320px;
      border-radius: 8px;
      background: #fdfefe;
      border: 1px solid #d8e1ea;
    }}
    .meta {{
      margin-top: 12px;
      font-size: 13px;
      color: #334a63;
    }}
    .hint {{
      margin-top: 6px;
      font-size: 12px;
      color: #5a7288;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="grid">
      <div class="card">
        <h2>Trajectory (Longitude vs Latitude)</h2>
        <svg id="traj"></svg>
      </div>
      <div class="card">
        <h2>Altitude over doc</h2>
        <svg id="alt"></svg>
      </div>
      <div class="card">
        <h2>Lat/Lon over doc</h2>
        <svg id="latlon"></svg>
      </div>
      <div class="card">
        <h2>Speed Proxy (Step Distance) over doc</h2>
        <svg id="speed"></svg>
      </div>
    </div>
    <div class="meta" id="meta"></div>
    <div class="hint">Trajectory 그래프에서 초록점은 시작, 빨간점은 끝입니다.</div>
  </div>
  <script>
    const data = {data_json};

    function minmax(arr) {{
      let min = Infinity;
      let max = -Infinity;
      for (const v of arr) {{
        if (v < min) min = v;
        if (v > max) max = v;
      }}
      if (min === max) {{
        const eps = Math.abs(min || 1) * 1e-6;
        return [min - eps, max + eps];
      }}
      return [min, max];
    }}

    function drawChart(svgId, series, opts = {{}}) {{
      const svg = document.getElementById(svgId);
      const w = 960;
      const h = 320;
      const pad = {{ l: 56, r: 16, t: 18, b: 42 }};
      svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);

      const xs = series.flatMap(s => s.points.map(p => p.x));
      const ys = series.flatMap(s => s.points.map(p => p.y));
      let [xmin, xmax] = minmax(xs);
      let [ymin, ymax] = minmax(ys);

      if (opts.equalAspect) {{
        const xr = xmax - xmin;
        const yr = ymax - ymin;
        const kx = xr / (w - pad.l - pad.r);
        const ky = yr / (h - pad.t - pad.b);
        const k = Math.max(kx, ky);
        const cx = 0.5 * (xmin + xmax);
        const cy = 0.5 * (ymin + ymax);
        const halfX = 0.5 * k * (w - pad.l - pad.r);
        const halfY = 0.5 * k * (h - pad.t - pad.b);
        xmin = cx - halfX;
        xmax = cx + halfX;
        ymin = cy - halfY;
        ymax = cy + halfY;
      }}

      const sx = x => pad.l + (x - xmin) * (w - pad.l - pad.r) / (xmax - xmin);
      const sy = y => h - pad.b - (y - ymin) * (h - pad.t - pad.b) / (ymax - ymin);
      const fmtX = opts.xFmt || (v => Number.isInteger(v) ? String(v) : v.toFixed(6));
      const fmtY = opts.yFmt || (v => Number.isInteger(v) ? String(v) : v.toFixed(6));

      let html = "";
      html += `<rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#fff"/>`;
      html += `<line x1="${{pad.l}}" y1="${{h-pad.b}}" x2="${{w-pad.r}}" y2="${{h-pad.b}}" stroke="#61758c" />`;
      html += `<line x1="${{pad.l}}" y1="${{pad.t}}" x2="${{pad.l}}" y2="${{h-pad.b}}" stroke="#61758c" />`;

      for (let i = 1; i <= 4; i++) {{
        const gx = pad.l + i * (w - pad.l - pad.r) / 5;
        const gy = pad.t + i * (h - pad.t - pad.b) / 5;
        html += `<line x1="${{gx}}" y1="${{pad.t}}" x2="${{gx}}" y2="${{h-pad.b}}" stroke="#edf1f5" />`;
        html += `<line x1="${{pad.l}}" y1="${{gy}}" x2="${{w-pad.r}}" y2="${{gy}}" stroke="#edf1f5" />`;
      }}

      for (const s of series) {{
        const d = s.points.map((p, i) => `${{i ? "L" : "M"}}${{sx(p.x).toFixed(2)}},${{sy(p.y).toFixed(2)}}`).join(" ");
        html += `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
      }}

      for (let i = 0; i <= 2; i++) {{
        const xVal = xmin + (xmax - xmin) * (i / 2);
        const xPx = sx(xVal);
        html += `<line x1="${{xPx}}" y1="${{h-pad.b}}" x2="${{xPx}}" y2="${{h-pad.b+5}}" stroke="#61758c" />`;
        html += `<text x="${{xPx}}" y="${{h-pad.b+18}}" text-anchor="middle" font-size="11" fill="#334a63">${{fmtX(xVal)}}</text>`;
      }}
      for (let i = 0; i <= 2; i++) {{
        const yVal = ymin + (ymax - ymin) * (i / 2);
        const yPx = sy(yVal);
        html += `<line x1="${{pad.l-5}}" y1="${{yPx}}" x2="${{pad.l}}" y2="${{yPx}}" stroke="#61758c" />`;
        html += `<text x="${{pad.l-8}}" y="${{yPx+4}}" text-anchor="end" font-size="11" fill="#334a63">${{fmtY(yVal)}}</text>`;
      }}
      html += `<text x="${{w * 0.5}}" y="${{h - 4}}" text-anchor="middle" font-size="12" fill="#334a63">${{opts.xLabel || ""}}</text>`;
      html += `<text x="16" y="${{h * 0.5}}" text-anchor="middle" font-size="12" fill="#334a63" transform="rotate(-90 16 ${{h * 0.5}})">${{opts.yLabel || ""}}</text>`;

      if (opts.markStartEnd && series.length > 0 && series[0].points.length > 1) {{
        const pts = series[0].points;
        const st = pts[0];
        const ed = pts[pts.length - 1];
        html += `<circle cx="${{sx(st.x)}}" cy="${{sy(st.y)}}" r="4.5" fill="#2ca02c"/>`;
        html += `<circle cx="${{sx(ed.x)}}" cy="${{sy(ed.y)}}" r="4.5" fill="#d62728"/>`;
        html += `<text x="${{sx(st.x)+7}}" y="${{sy(st.y)-7}}" font-size="11" fill="#2a6f2f">start</text>`;
        html += `<text x="${{sx(ed.x)+7}}" y="${{sy(ed.y)-7}}" font-size="11" fill="#8e1e1e">end</text>`;
      }}

      if (opts.legend) {{
        html += `<rect x="${{w-178}}" y="12" width="160" height="${{series.length * 18 + 12}}" fill="#ffffff" stroke="#d7e0e9" rx="6" />`;
        let y0 = 26;
        for (const s of series) {{
          html += `<line x1="${{w-170}}" y1="${{y0}}" x2="${{w-145}}" y2="${{y0}}" stroke="${{s.color}}" stroke-width="3"/>`;
          html += `<text x="${{w-140}}" y="${{y0+4}}" font-size="12" fill="#23384f">${{s.name}}</text>`;
          y0 += 18;
        }}
      }}

      svg.innerHTML = html;
    }}

    const traj = data.map(d => ({{ x: d.lon, y: d.lat }}));
    const alt = data.map(d => ({{ x: d.doc, y: d.alt }}));
    const lat = data.map(d => ({{ x: d.doc, y: d.lat }}));
    const lon = data.map(d => ({{ x: d.doc, y: d.lon }}));
    const step = data.map(d => ({{ x: d.doc, y: d.step_m }}));

    drawChart("traj", [{{ name: "path", color: "#1f77b4", points: traj }}], {{
      xLabel: "Longitude",
      yLabel: "Latitude",
      equalAspect: true,
      xFmt: v => v.toFixed(6),
      yFmt: v => v.toFixed(6),
      markStartEnd: true,
      legend: false
    }});
    drawChart("alt", [{{ name: "alt_m", color: "#ff7f0e", points: alt }}], {{
      xLabel: "doc",
      yLabel: "alt_m",
      xFmt: v => Math.round(v).toString(),
      yFmt: v => v.toFixed(3),
      legend: false
    }});
    drawChart("latlon", [
      {{ name: "lat_deg", color: "#2ca02c", points: lat }},
      {{ name: "lon_deg", color: "#d62728", points: lon }}
    ], {{
      xLabel: "doc",
      yLabel: "degree",
      xFmt: v => Math.round(v).toString(),
      yFmt: v => v.toFixed(6),
      legend: true
    }});
    drawChart("speed", [{{ name: "step_distance_m", color: "#9467bd", points: step }}], {{
      xLabel: "doc",
      yLabel: "meter/step",
      xFmt: v => Math.round(v).toString(),
      yFmt: v => v.toFixed(3),
      legend: false
    }});

    const docs = data.map(d => d.doc);
    const alts = data.map(d => d.alt);
    const lats = data.map(d => d.lat);
    const lons = data.map(d => d.lon);
    const steps = data.map(d => d.step_m);
    const cums = data.map(d => d.cum_m);
    const [docMin, docMax] = minmax(docs);
    const [altMin, altMax] = minmax(alts);
    const [latMin, latMax] = minmax(lats);
    const [lonMin, lonMax] = minmax(lons);
    const [stepMin, stepMax] = minmax(steps);
    const [cumMin, cumMax] = minmax(cums);
    document.getElementById("meta").textContent =
      `rows=${{data.length}}, doc=[${{docMin}}..${{docMax}}], ` +
      `lat=[${{latMin.toFixed(8)}}..${{latMax.toFixed(8)}}], ` +
      `lon=[${{lonMin.toFixed(8)}}..${{lonMax.toFixed(8)}}], ` +
      `alt=[${{altMin.toFixed(3)}}..${{altMax.toFixed(3)}}], ` +
      `step_m=[${{stepMin.toFixed(3)}}..${{stepMax.toFixed(3)}}], ` +
      `cum_m=[${{cumMin.toFixed(1)}}..${{cumMax.toFixed(1)}}]`;
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = add_motion_fields(load_rows(csv_path))
    out_path = args.out or csv_path.with_name(f"{csv_path.stem}_gps.html")
    out_path.write_text(build_html(csv_path.name, rows), encoding="utf-8")
    print(f"Saved plot: {out_path}")
    if args.save_png:
        png_path = args.png_out or csv_path.with_name(f"{csv_path.stem}_gps.png")
        render_png(rows, png_path, csv_path.name)
        print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
