#!/usr/bin/env python3
"""Download a public MEGA file or folder link with nothing but Python 3 and the openssl command
(both are on practically every Linux server), so no installation is needed.

    python3 mega_get.py "https://mega.nz/file/ID#KEY"   [target_dir]
    python3 mega_get.py "https://mega.nz/folder/ID#KEY" [target_dir]

Keep the link in quotes (it contains #). Folders are downloaded with their sub-folders.
"""
import base64
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request

API = "https://g.api.mega.co.nz/cs"


def b64(s):
    s = s.replace("-", "+").replace("_", "/").replace(",", "")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def a32(b):
    return struct.unpack(">%dI" % (len(b) // 4), b)


def raw(words):
    return struct.pack(">%dI" % len(words), *words)


def openssl(mode, key, data, iv=None):
    cmd = ["openssl", "enc", "-d", "-" + mode, "-K", key.hex(), "-nopad"] + (["-iv", iv.hex()] if iv else [])
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def api(req, folder=None):
    url = API + "?id=1" + (f"&n={folder}" if folder else "")
    body = json.dumps([req]).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=60) as r:
        res = json.loads(r.read())
    res = res[0] if isinstance(res, list) else res
    if isinstance(res, int):
        sys.exit(f"MEGA refused the request (error {res}); is the link complete and still shared?")
    return res


def file_key(words):
    """8-word node key -> (AES key, CTR IV)."""
    return raw([words[i] ^ words[i + 4] for i in range(4)]), raw([words[4], words[5], 0, 0])


def name_of(attr_b64, key):
    plain = openssl("aes-128-cbc", key, b64(attr_b64), b"\0" * 16).rstrip(b"\0")
    if not plain.startswith(b"MEGA"):
        sys.exit("could not decrypt the file name: the key part of the link (after #) is wrong")
    return json.loads(plain[4:].decode("utf-8", "replace"))["n"]


def fetch(url, key, iv, path, size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".part", "wb") as out:
        dec = subprocess.Popen(["openssl", "enc", "-d", "-aes-128-ctr", "-K", key.hex(), "-iv", iv.hex()],
                               stdin=subprocess.PIPE, stdout=out)
        done = 0
        with urllib.request.urlopen(url, timeout=120) as r:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                dec.stdin.write(chunk)
                done += len(chunk)
                print(f"\r  {path}: {done / 1e6:.1f} / {size / 1e6:.1f} MB", end="", flush=True)
        dec.stdin.close()
        if dec.wait() != 0:
            sys.exit("\nopenssl failed to decrypt")
    os.replace(path + ".part", path)
    print()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    link, target = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
    m = re.search(r"/(file|folder)/([^#/?]+)#([^/?]+)", link)
    if m:
        kind, ident, keystr = m.groups()
    else:
        m = re.search(r"#(F?)!([^!]+)!([^!/?]+)", link)        # old style #!id!key and #F!id!key
        if not m:
            sys.exit("not a MEGA file or folder link")
        kind, ident, keystr = ("folder" if m.group(1) else "file"), m.group(2), m.group(3)
    if subprocess.run(["openssl", "version"], capture_output=True).returncode != 0:
        sys.exit("the openssl command is needed (it is on almost every Linux system)")

    if kind == "file":
        key, iv = file_key(a32(b64(keystr)))
        info = api({"a": "g", "g": 1, "ssl": 2, "p": ident})
        fetch(info["g"], key, iv, os.path.join(target, name_of(info["at"], key)), info["s"])
        return

    share_key = b64(keystr)[:16]
    nodes = api({"a": "f", "c": 1, "r": 1, "ca": 1}, folder=ident)["f"]
    handles = {n["h"] for n in nodes}
    tree = {}
    for n in nodes:
        parts = [p.split(":", 1) for p in n["k"].split("/") if ":" in p]
        enc = next((k for owner, k in parts if owner == ident or owner in handles), parts[0][1])
        words = a32(openssl("aes-128-ecb", share_key, b64(enc)))
        if n["t"] == 0:
            key, iv = file_key(words)
        else:
            key, iv = raw(words[:4]), None
        tree[n["h"]] = dict(node=n, key=key, iv=iv, name=name_of(n["a"], key), parent=n.get("p"))

    def path_of(h):
        parts = []
        while h in tree:
            parts.append(tree[h]["name"])
            h = tree[h]["parent"]
        return os.path.join(target, *reversed(parts))

    files = [t for t in tree.values() if t["node"]["t"] == 0]
    print(f"{len(files)} file(s), {sum(t['node']['s'] for t in files) / 1e6:.1f} MB")
    for t in files:
        path = path_of(t["node"]["h"])
        if os.path.exists(path) and os.path.getsize(path) == t["node"]["s"]:
            print(f"  {path}: already there")
            continue
        info = api({"a": "g", "g": 1, "ssl": 2, "n": t["node"]["h"]}, folder=ident)
        fetch(info["g"], t["key"], t["iv"], path, t["node"]["s"])


if __name__ == "__main__":
    main()
