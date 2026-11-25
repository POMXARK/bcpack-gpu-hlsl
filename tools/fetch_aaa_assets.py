#!/usr/bin/env python3
"""Downloads open-source CC0 / CC-BY 4.0 AAA 3D assets (Khronos CarConcept.glb + Poly Haven 2K/4K PBR & HDRI)
into aaa_assets/ for the bcpack-gpu-hlsl AAA Showcase."""

import argparse
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent / "aaa_assets"
UA = {"User-Agent": "bcpack-gpu-hlsl-bench/0.8"}


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name} ({dst.stat().st_size // 1024} KiB)")
        return
    print(f"[get ] {url} -> {dst.name}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    dst.write_bytes(data)
    print(f"[done] {dst.name} ({len(data) // 1024} KiB)")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def save_png(src_img: Path, dst_png: Path, dim: int) -> None:
    if dst_png.exists() and dst_png.stat().st_size > 0:
        print(f"[skip] {dst_png.name} ({dim}x{dim} PNG cached)")
        return
    with Image.open(src_img) as im:
        im = im.convert("RGBA")
        if im.size != (dim, dim):
            im = im.resize((dim, dim), Image.Resampling.LANCZOS)
        im.save(dst_png, format="PNG", compress_level=3)
    print(f"[pack] {dst_png.name} ({dim}x{dim} PNG, {dst_png.stat().st_size // 1024} KiB)")


def read_radiance_hdr(hdr_path: Path) -> np.ndarray:
    """Decodes standard RLE Radiance RGBE (.hdr) file into float32 (H, W, 3) linear RGB."""
    raw = hdr_path.read_bytes()
    idx = 0
    # Parse header until empty line
    while idx < len(raw):
        nl = raw.find(b"\n", idx)
        if nl == -1:
            raise ValueError("Invalid HDR header")
        line = raw[idx:nl].decode("ascii", errors="ignore").strip()
        idx = nl + 1
        if line == "":
            break
    # Parse resolution line: e.g. "-Y 1024 +X 2048"
    nl = raw.find(b"\n", idx)
    res_line = raw[idx:nl].decode("ascii").strip().split()
    idx = nl + 1
    if len(res_line) != 4 or res_line[0] != "-Y" or res_line[2] != "+X":
        raise ValueError(f"Unsupported HDR orientation: {res_line}")
    h = int(res_line[1])
    w = int(res_line[3])

    img = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        if raw[idx] != 2 or raw[idx + 1] != 2:
            raise ValueError("Expected new-style RLE scanline in HDR")
        scan_w = (raw[idx + 2] << 8) | raw[idx + 3]
        if scan_w != w:
            raise ValueError(f"Scanline width mismatch: {scan_w} != {w}")
        idx += 4
        for c in range(4):
            x = 0
            while x < w:
                code = raw[idx]
                idx += 1
                if code > 128:
                    run = code - 128
                    val = raw[idx]
                    idx += 1
                    img[y, x : x + run, c] = val
                    x += run
                else:
                    run = code
                    img[y, x : x + run, c] = np.frombuffer(raw, dtype=np.uint8, count=run, offset=idx)
                    idx += run
                    x += run

    e = img[:, :, 3].astype(np.int32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    nonzero = e > 0
    scale = np.zeros((h, w), dtype=np.float32)
    scale[nonzero] = np.ldexp(1.0, e[nonzero] - (128 + 8))
    rgb[:, :, 0] = (img[:, :, 0].astype(np.float32) + 0.5) * scale
    rgb[:, :, 1] = (img[:, :, 1].astype(np.float32) + 0.5) * scale
    rgb[:, :, 2] = (img[:, :, 2].astype(np.float32) + 0.5) * scale
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch open CC0/CC-BY 4.0 AAA benchmark assets")
    parser.add_argument("--no-4k", action="store_true", help="Skip 4K Poly Haven PNG generation")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "_raw"
    tmp_dir.mkdir(exist_ok=True)

    # 1) Khronos CarConcept.glb (CC-BY 4.0)
    download_file(
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CarConcept/glTF-Binary/CarConcept.glb",
        OUT_DIR / "CarConcept.glb",
    )

    # 2) Poly Haven CC0 2K + 4K PBR texture sets
    ph_sets = [
        ("asphalt_pit_lane", "asphalt"),
        ("brick_pavement", "promenade"),
        ("concrete_wall_008", "concrete"),
    ]
    for ph_id, prefix in ph_sets:
        files = fetch_json(f"https://api.polyhaven.com/files/{ph_id}")
        for map_key, out_suffix in [("Diffuse", "albedo"), ("nor_dx", "normal"), ("arm", "arm")]:
            k2 = files[map_key]["2k"]
            ext2 = "jpg" if "jpg" in k2 else "png"
            raw_path2 = tmp_dir / f"{ph_id}_{map_key}_2k.{ext2}"
            download_file(k2[ext2]["url"], raw_path2)
            save_png(raw_path2, OUT_DIR / f"{prefix}_{out_suffix}.png", 2048)

            if not args.no_4k and "4k" in files[map_key]:
                k4 = files[map_key]["4k"]
                ext4 = "jpg" if "jpg" in k4 else "png"
                raw_path4 = tmp_dir / f"{ph_id}_{map_key}_4k.{ext4}"
                download_file(k4[ext4]["url"], raw_path4)
                save_png(raw_path4, OUT_DIR / f"{prefix}_{out_suffix}_4k.png", 4096)

    # 3) Poly Haven CC0 2K Sunset HDRI -> raw 2048x1024 RGBA16F (.fp16)
    hdri_fp16 = OUT_DIR / "sunset_hdri_2k.fp16"
    if not hdri_fp16.exists():
        h_files = fetch_json("https://api.polyhaven.com/files/belfast_sunset_puresky")
        hdr_url = h_files["hdri"]["2k"]["hdr"]["url"]
        hdr_raw = tmp_dir / "belfast_sunset_puresky_2k.hdr"
        download_file(hdr_url, hdr_raw)
        rgb = read_radiance_hdr(hdr_raw)
        h, w, _ = rgb.shape
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = np.clip(rgb, 0.0, 65000.0)
        fp16 = rgba.astype(np.float16)
        # Header: magic b"HDRF" + u32 width + u32 height, followed by raw little-endian float16 RGBA
        with open(hdri_fp16, "wb") as f:
            f.write(b"HDRF")
            f.write(struct.pack("<II", w, h))
            f.write(fp16.tobytes())
        print(f"[pack] {hdri_fp16.name} ({w}x{h} FP16, {hdri_fp16.stat().st_size // 1024} KiB)")

    # 4) Attribution notices
    (OUT_DIR / "ATTRIBUTION.txt").write_text(
        "Open AAA Benchmark Assets (downloaded via tools/fetch_aaa_assets.py):\n"
        "- CarConcept.glb: KhronosGroup/glTF-Sample-Assets, Copyright 2024 Darmstadt Graphics Group GmbH (CC-BY 4.0)\n"
        "- asphalt_pit_lane, brick_pavement, concrete_wall_008, belfast_sunset_puresky HDRI: Poly Haven (CC0 1.0 Public Domain)\n",
        encoding="utf-8",
    )
    print("All open AAA assets ready in:", OUT_DIR)


if __name__ == "__main__":
    main()
