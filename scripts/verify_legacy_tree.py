"""Verify the pre-pivot root is preserved by tree identity, not just matching prose."""
import subprocess

EXPECTED = "d79ce6d0b4251b51d8b1fd8d099d89b55ba1329c"
actual = subprocess.check_output(
    ["git", "rev-parse", "HEAD:legacy"], text=True
).strip()
if actual != EXPECTED:
    raise SystemExit(f"legacy tree mismatch: {actual} != {EXPECTED}")
kind = subprocess.check_output(["git", "cat-file", "-t", actual], text=True).strip()
if kind != "tree":
    raise SystemExit(f"legacy object was not a tree: {kind}")
print(f"Immutable pre-pivot root tree confirmed: {actual}")
