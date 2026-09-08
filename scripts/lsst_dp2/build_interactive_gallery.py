"""Build a self-contained two-slider HTML viewer for DP2 gallery PNGs."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

MAG_LABELS = ("18 <= i < 20", "20 <= i < 21", "21 <= i < 22", "22 <= i < 23", "23 <= i < 24")
SIZE_LABELS = (
    '0.4" <= Re < 0.6"',
    '0.6" <= Re < 1.0"',
    '1.0" <= Re < 1.5"',
    'Re >= 1.5"',
)
FILE_PATTERN = re.compile(
    r"mag_(18_20|20_21|21_22|22_23|23_24)__"
    r"reff_(0p4_0p6|0p6_1|1_1p5|1p5_inf)__bands\.png"
)
MAG_KEYS = ("18_20", "20_21", "21_22", "22_23", "23_24")
SIZE_KEYS = ("0p4_0p6", "0p6_1", "1_1p5", "1p5_inf")


def collect_images(input_dir: Path) -> dict[str, str]:
    """Return all 20 gallery images as embedded PNG data URLs."""
    images = {}
    for path in sorted(input_dir.glob("*.png")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            mag_index = MAG_KEYS.index(match.group(1))
            size_index = SIZE_KEYS.index(match.group(2))
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            images[f"{mag_index}-{size_index}"] = f"data:image/png;base64,{encoded}"
    expected = {f"{mi}-{si}" for mi in range(5) for si in range(4)}
    missing = sorted(expected - images.keys())
    if missing:
        raise ValueError(f"missing magnitude-size gallery cells: {missing}")
    return images


def build_html(images: dict[str, str]) -> str:
    """Return the standalone interactive gallery document."""
    image_json = json.dumps(images, separators=(",", ":"))
    mag_json = json.dumps(MAG_LABELS)
    size_json = json.dumps(SIZE_LABELS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LSST DP2 magnitude-size gallery</title>
<style>
  :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
  body {{ margin: 0; background: #f7f7f7; color: #171717; }}
  main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
  h1 {{ margin: 0 0 14px; font-size: 22px; letter-spacing: 0; }}
  .controls {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 16px; }}
  label {{ display: grid; gap: 7px; font-size: 14px; font-weight: 600; }}
  output {{ font-variant-numeric: tabular-nums; font-weight: 500; }}
  input[type="range"] {{ width: 100%; accent-color: #167a72; }}
  figure {{ margin: 0; background: white; border: 1px solid #d5d5d5; border-radius: 6px; overflow: hidden; }}
  img {{ display: block; width: 100%; height: auto; }}
  figcaption {{ padding: 10px 12px; font-size: 13px; border-top: 1px solid #e5e5e5; }}
  @media (max-width: 700px) {{ .controls {{ grid-template-columns: 1fr; gap: 12px; }} }}
</style>
</head>
<body>
<main>
  <h1>LSST DP2 MMU cutouts: magnitude x effective radius</h1>
  <section class="controls">
    <label>i-band cModel magnitude <output id="mag-label"></output>
      <input id="mag" type="range" min="0" max="4" step="1" value="0">
    </label>
    <label>Sersic major-axis effective radius <output id="size-label"></output>
      <input id="size" type="range" min="0" max="3" step="1" value="0">
    </label>
  </section>
  <figure>
    <img id="gallery" alt="Full 160 by 160 pixel ugrizy cutout gallery">
    <figcaption id="caption"></figcaption>
  </figure>
</main>
<script>
const images = {image_json};
const magLabels = {mag_json};
const sizeLabels = {size_json};
const mag = document.getElementById("mag");
const size = document.getElementById("size");
function render() {{
  const mi = Number(mag.value);
  const si = Number(size.value);
  document.getElementById("mag-label").value = magLabels[mi];
  document.getElementById("size-label").value = sizeLabels[si];
  document.getElementById("gallery").src = images[`${{mi}}-${{si}}`];
  document.getElementById("caption").textContent =
    `${{magLabels[mi]}} | ${{sizeLabels[si]}} | ugrizy, 160 x 160 pixels`;
}}
mag.addEventListener("input", render);
size.addEventListener("input", render);
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    images = collect_images(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(images), encoding="ascii")
    print(f"Wrote {len(images)} cells to {args.output}")


if __name__ == "__main__":
    main()
