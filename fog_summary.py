"""Summarise a fog-sweep run: one row per obstacle, fog level and distance (from frames.csv)."""
import csv, json, math, sys, os
out = sys.argv[1] if len(sys.argv) > 1 else "out/fog_validation"
rows = [r for r in csv.DictReader(open(os.path.join(out, "frames.csv"))) if r["background"] in ("False", "false", "0", "")]
with open(os.path.join(out, "fog_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "fog_percent", "fog_density_per_m", "fog_visibility_m", "distance_m", "dist_cam_m", "fog_transmittance", "labelled", "visible_px", "bbox_w", "bbox_h", "image"])
    for r in sorted(rows, key=lambda r: (r["case"], float(r["fog_density"] or 0), float(r["milestone_m"] or 0))):
        d = float(r["fog_density"] or 0)
        pct = round(d / 0.026 * 100)
        vis = round(3.912 / d) if d else ""
        bw = bh = ""
        if r["bbox_visible_x0"]:
            bw = round(float(r["bbox_visible_x1"]) - float(r["bbox_visible_x0"]), 1)
            bh = round(float(r["bbox_visible_y1"]) - float(r["bbox_visible_y0"]), 1)
        w.writerow([r["case"], pct, d, vis, r["milestone_m"], r["dist_euclid_cam_m"], round(math.exp(-d * float(r["dist_euclid_cam_m"])), 4), r["labelled"], r["visible_px"], bw, bh, r["image"]])
print("wrote", os.path.join(out, "fog_summary.csv"), len(rows), "rows")
