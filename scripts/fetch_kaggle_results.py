"""Decode the base64 result tarball that a Kaggle kernel printed into its log.
Input: the JSON returned by the Kaggle API `list_notebook_session_output` (saved to a file), with a
"log" field holding the kernel's stdout/stderr stream. Output: files extracted under the repo root.
python scripts/fetch_kaggle_results.py session_output.json [dest_dir]"""
import base64
import hashlib
import io
import json
import sys
import tarfile

src = sys.argv[1]
dest = sys.argv[2] if len(sys.argv) > 2 else "."
d = json.load(open(src))
log = json.loads(d["log"]) if isinstance(d["log"], str) else d["log"]
text = "".join(e["data"] for e in log if e.get("stream_name") == "stdout")
lines = text.splitlines()
i0 = next(i for i, l in enumerate(lines) if l.startswith("RESULTS_B64_BEGIN"))
sha = lines[i0].split()[1]
i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith("RESULTS_B64_END"))
blob = "".join(lines[i0 + 1:i1]).strip()
assert hashlib.sha256(blob.encode()).hexdigest() == sha, "checksum mismatch"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tf:
    names = tf.getnames()
    assert all(not n.startswith("/") and ".." not in n for n in names)
    tf.extractall(dest, filter="data")
print(f"extracted {len(names)} entries to {dest}; checksum ok")
# also save the human-readable part of the log (without the blob)
open(src.replace(".json", "") + "_log.txt", "w").write("\n".join(lines[:i0] + lines[i1 + 1:]))
