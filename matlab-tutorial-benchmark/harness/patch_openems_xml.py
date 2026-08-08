#!/usr/bin/env python3
"""Set deterministic benchmark termination controls without rewriting XML."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def set_attribute(tag: str, name: str, value: str) -> str:
    pattern = re.compile(rf"(\s{name}\s*=\s*)([\"']).*?\2", re.IGNORECASE)
    if pattern.search(tag):
        return pattern.sub(lambda match: f"{match.group(1)}\"{value}\"", tag, count=1)
    return tag[:-1] + f' {name}="{value}">'


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: patch_openems_xml.py XML_FILE TIMESTEPS", file=sys.stderr)
        return 2

    xml_path = Path(sys.argv[1])
    timesteps = int(sys.argv[2])
    xml_text = xml_path.read_text(encoding="utf-8")
    match = re.search(r"<FDTD\b[^>]*>", xml_text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"no FDTD element found in {xml_path}")

    fdtd_tag = set_attribute(match.group(0), "NumberOfTimesteps", str(timesteps))
    fdtd_tag = set_attribute(fdtd_tag, "endCriteria", "0")
    patched = xml_text[: match.start()] + fdtd_tag + xml_text[match.end() :]
    xml_path.write_text(patched, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
