#!/usr/bin/env python3
"""Build bench_x86_64 in ReleaseFast for each safety-layer configuration into bench/bins/."""
import os, shutil, subprocess

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(root, "bench/bins"), exist_ok=True)
for name, v, h in (("off", "false", "false"), ("validate", "true", "false"), ("validate_hooks", "true", "true")):
    subprocess.run(["mise", "exec", "--", "zig", "build", "-Doptimize=ReleaseFast", f"-Dvalidate={v}", f"-Dhooks={h}"], cwd=root, check=True)
    shutil.copy(os.path.join(root, "zig-out/bin/bench_x86_64"), os.path.join(root, "bench/bins", name))
    print("built", name)
