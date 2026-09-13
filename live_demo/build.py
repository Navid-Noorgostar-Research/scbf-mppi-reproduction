"""Inline sim.js into demo_src.html -> demo_artifact.html (fragment for the Artifact tool) and
safety_bench.html (full standalone document, offline, for the package)."""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "demo_src.html"), encoding="utf-8").read()
sim = open(os.path.join(here, "sim.js"), encoding="utf-8").read()
frag = src.replace("/*__SIM_JS__*/", sim)
open(os.path.join(here, "demo_artifact.html"), "w", encoding="utf-8").write(frag)
full = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        + frag.split("<div class=\"wrap\">")[0] + "</head><body>\n<div class=\"wrap\">" + frag.split("<div class=\"wrap\">", 1)[1] + "\n</body></html>\n")
open(os.path.join(here, "safety_bench.html"), "w", encoding="utf-8").write(full)
print("built", len(frag), len(full))
