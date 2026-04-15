#!/usr/bin/env python3
"""
Downloader for the Nature 2023 "Photos and rendered images of LEGO bricks" corpus.

Paper:  https://www.nature.com/articles/s41597-023-02682-2
Host:   mostwiedzy.pl (Gdansk University of Technology institutional repo)

The full corpus is ~150k real photos + ~1.5M renders across 5 datasets;
each dataset is a single `.zip` registered under a DOI.

Because we already sit on a 268k+ render corpus from the local Blender run
(~/Desktop/synthetic_dataset), the PRIMARY VALUE of this download is the
REAL-PHOTO portion, which closes the sim-to-real gap. This script prioritises
the real-photo datasets first and defers the big render zips.

Output layout (matches what `train_contrastive.py` + `train_yolo.py` expect):

  ml/data/nature_2023/
    classification/
      <part_num>/*.jpg       (real photos + renders, 447 classes)
    detection/
      images/*.jpg
      annotations/*.xml      (PASCAL VOC)
      labels/*.txt           (YOLO, auto-converted)

Resumability: each dataset's zip is checkpointed by name + byte offset.
Interrupted downloads resume via HTTP Range.

Usage:
    python3 ml/data/download_nature_2023.py --dest ml/data/nature_2023 [--priority real|renders|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("ERROR: `requests` not installed. Run: pip install requests tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: `tqdm` not installed. Run: pip install tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("nature_2023")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry. These are the dataset pages on mostwiedzy.pl. The actual
# .zip download URL is exposed either via the page's DataCite metadata OR the
# page exposes a button. We probe each page for the download link at runtime.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NatureDataset:
    name: str
    priority: str              # "real" or "renders" — real-photo sets first
    page_url: str
    target_subdir: str         # relative path under dest
    description: str


DATASETS: List[NatureDataset] = [
    NatureDataset(
        name="tagged_images_bounding_boxes",
        priority="real",
        page_url="https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0",
        target_subdir="detection",
        description="2,933 real photos + 2,908 renders with PASCAL VOC XML bounding boxes",
    ),
    NatureDataset(
        name="images_of_lego_bricks",
        priority="real",
        page_url="https://mostwiedzy.pl/en/open-research-data/images-of-lego-bricks,202309140837142278781-0",
        target_subdir="classification_real",
        description="Real photographs of LEGO bricks (classification)",
    ),
    NatureDataset(
        name="classification_network",
        priority="renders",
        page_url="https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0",
        target_subdir="classification_renders",
        description="447 classes; 52k photos + 567k renders (bulk)",
    ),
    NatureDataset(
        name="conveyor_video",
        priority="renders",
        page_url="https://mostwiedzy.pl/en/open-research-data-series/video-of-lego-bricks-on-conveyor-belt,202011132226557715481-0/catalog",
        target_subdir="conveyor",
        description="Video of bricks on conveyor — optional, useful for temporal aug",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Link discovery
# ─────────────────────────────────────────────────────────────────────────────

_USER_AGENT = "BrickScan-DataFetcher/1.0 (+https://github.com/edrickchang13/BrickScan)"


def discover_zip_url(page_url: str, session: requests.Session) -> Optional[str]:
    """
    Probe a mostwiedzy.pl dataset page for the downloadable archive URL.

    MostWiedzy embeds the file URL in a JSON-LD block AND in direct <a> links
    to the `/files/` endpoint. We try the cheaper <a href> scrape first.
    """
    try:
        r = session.get(page_url, timeout=30)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        # Common when running in a sandboxed environment that blocks mostwiedzy.pl.
        # Print actionable manual-download instructions instead of just warning.
        log.warning(
            "Could not reach mostwiedzy.pl — you may be on a restricted network.\n"
            "  Manual download steps:\n"
            "    1. Open this URL in a browser on an unrestricted network:\n"
            "       %s\n"
            "    2. Click the 'Download' button on the dataset page.\n"
            "    3. Save the .zip into `ml/data/nature_2023/<target_subdir>/`\n"
            "    4. Re-run this script — it will detect the zip and extract it.\n"
            "  Underlying error: %s",
            page_url, e,
        )
        return None
    except requests.RequestException as e:
        log.warning("Could not fetch page %s: %s", page_url, e)
        return None

    html = r.text

    # Direct file download link pattern (most common on mostwiedzy)
    # e.g. /open-research-data/.../file/<uuid>/download or .zip links
    patterns = [
        r'href="(https?://mostwiedzy\.pl/[^"]+?\.zip)"',
        r'href="(/open-research-data/[^"]+?\.zip)"',
        r'href="(https?://mostwiedzy\.pl/[^"]+?/download[^"]*)"',
        r'"contentUrl"\s*:\s*"([^"]+?\.zip)"',
    ]
    for pat in patterns:
        for match in re.finditer(pat, html):
            url = match.group(1)
            if url.startswith("/"):
                url = "https://mostwiedzy.pl" + url
            log.info("Discovered archive URL: %s", url)
            return url

    log.warning("No .zip URL found on page %s — manual download required.", page_url)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Resumable download
# ─────────────────────────────────────────────────────────────────────────────

def download_with_resume(url: str, dest_path: Path, session: requests.Session) -> bool:
    """
    Download `url` to `dest_path`, resuming if a partial file already exists.

    Returns True on success. On HTTP-level failure, leaves the partial file
    intact so a later run can resume.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"User-Agent": _USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        log.info("Resuming %s from byte %d", dest_path.name, existing)

    try:
        with session.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code == 416:
                # Range unsatisfiable — file is already complete
                log.info("%s already fully downloaded (server says 416)", dest_path.name)
                tmp_path.rename(dest_path)
                return True
            r.raise_for_status()

            total = existing
            if "Content-Length" in r.headers:
                total += int(r.headers["Content-Length"])

            mode = "ab" if existing > 0 else "wb"
            with open(tmp_path, mode) as f, tqdm(
                total=total,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        tmp_path.rename(dest_path)
        return True
    except requests.RequestException as e:
        log.error("Download failed: %s — partial file preserved at %s", e, tmp_path)
        return False
    except KeyboardInterrupt:
        log.warning("Interrupted — partial file preserved at %s", tmp_path)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Extraction + PASCAL VOC → YOLO conversion
# ─────────────────────────────────────────────────────────────────────────────

def extract_archive(zip_path: Path, target_dir: Path) -> bool:
    """Extract a zip; skip if target already looks populated (> 50 files)."""
    if target_dir.exists() and sum(1 for _ in target_dir.rglob("*")) > 50:
        log.info("Target %s already has files — skipping extraction.", target_dir)
        return True
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            log.info("Extracting %d members from %s → %s", len(members), zip_path.name, target_dir)
            for m in tqdm(members, desc=f"Unzipping {zip_path.name}"):
                zf.extract(m, target_dir)
        return True
    except zipfile.BadZipFile as e:
        log.error("Bad zip %s: %s — delete and re-download.", zip_path, e)
        return False


def convert_voc_to_yolo(detection_dir: Path) -> int:
    """
    Walk any PASCAL VOC .xml annotation files under `detection_dir`, write a
    matching YOLO-format .txt alongside with class=0, normalised coords.
    Returns number of conversions.
    """
    # Single class for all LEGO bricks ("lego_piece") — matches train_yolo.py
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        return 0

    annot_root = None
    for candidate in ("annotations", "Annotations", "voc", "xml"):
        p = detection_dir / candidate
        if p.is_dir():
            annot_root = p
            break
    if annot_root is None:
        # Flat layout — search for *.xml anywhere under detection_dir
        annot_root = detection_dir

    labels_root = detection_dir / "labels"
    labels_root.mkdir(exist_ok=True)
    converted = 0

    for xml_path in annot_root.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                continue
            img_w = int(size.findtext("width") or 0) or 1
            img_h = int(size.findtext("height") or 0) or 1

            lines = []
            for obj in root.findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.findtext("xmin") or 0)
                ymin = float(bb.findtext("ymin") or 0)
                xmax = float(bb.findtext("xmax") or 0)
                ymax = float(bb.findtext("ymax") or 0)
                cx = ((xmin + xmax) / 2.0) / img_w
                cy = ((ymin + ymax) / 2.0) / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                continue
            txt_path = labels_root / (xml_path.stem + ".txt")
            txt_path.write_text("\n".join(lines))
            converted += 1
        except Exception as e:
            log.debug("Skipping malformed %s: %s", xml_path, e)

    log.info("Converted %d PASCAL VOC XML → YOLO txt in %s", converted, labels_root)
    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(dest: Path, priority: str, datasets: List[NatureDataset]) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / ".download_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    selected = [d for d in datasets if priority == "all" or d.priority == priority]
    if not selected:
        log.warning("No datasets match priority=%s", priority)
        return 1

    failures = 0
    for ds in selected:
        log.info("=" * 70)
        log.info("Dataset: %s — %s", ds.name, ds.description)
        log.info("Page:    %s", ds.page_url)

        target_dir = dest / ds.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Check for a pre-placed zip (user downloaded it manually via a browser
        #    because mostwiedzy.pl serves a JS-challenge interstitial that plain
        #    requests can't pass). Any *.zip in the target dir works.
        pre_placed = sorted(target_dir.glob("*.zip"))
        if pre_placed:
            zip_path = pre_placed[0]
            log.info("Found pre-placed zip: %s (%.1f MB)",
                     zip_path.name, zip_path.stat().st_size / 1e6)
            if not extract_archive(zip_path, target_dir):
                failures += 1
                continue
            if ds.target_subdir == "detection":
                convert_voc_to_yolo(target_dir)
            manifest[ds.name] = {
                "url": "pre-placed",
                "zip_path": str(zip_path),
                "target_dir": str(target_dir),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue

        # 2. Try auto-discovery of the zip URL from the dataset page.
        zip_url = manifest.get(ds.name, {}).get("url") or discover_zip_url(ds.page_url, session)
        if not zip_url or zip_url == "pre-placed":
            log.error(
                "Skipping %s — no pre-placed zip and could not auto-discover URL.\n"
                "  MANUAL DOWNLOAD STEPS:\n"
                "    1. Open this URL in Safari (clicks through the JS challenge):\n"
                "       %s\n"
                "    2. Click the 'Download' button on the page.\n"
                "    3. Move the downloaded .zip into:\n"
                "       %s/\n"
                "    4. Re-run this script — it'll detect the zip and extract it.",
                ds.name, ds.page_url, target_dir,
            )
            failures += 1
            continue

        zip_name = Path(urlparse(zip_url).path).name or f"{ds.name}.zip"
        zip_path = target_dir / zip_name

        if not zip_path.exists():
            ok = download_with_resume(zip_url, zip_path, session)
            if not ok:
                failures += 1
                continue
        else:
            log.info("Already downloaded: %s", zip_path.name)

        if not extract_archive(zip_path, target_dir):
            failures += 1
            continue

        if ds.target_subdir == "detection":
            convert_voc_to_yolo(target_dir)

        manifest[ds.name] = {
            "url": zip_url,
            "zip_path": str(zip_path),
            "target_dir": str(target_dir),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("=" * 70)
    log.info("Done. %d dataset(s) failed.", failures)
    log.info("Manifest: %s", manifest_path)
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("ml/data/nature_2023"),
                        help="Destination root directory (default: ml/data/nature_2023)")
    parser.add_argument("--priority", choices=["real", "renders", "all"], default="real",
                        help="Which datasets to download first. 'real' = bounding-box + real photos only (default, smaller, highest value). 'renders' = the big 447-class zip. 'all' = everything.")
    args = parser.parse_args()

    dest = args.dest.resolve()
    log.info("Nature 2023 downloader — dest=%s priority=%s", dest, args.priority)
    return run(dest, args.priority, DATASETS)


if __name__ == "__main__":
    sys.exit(main())
