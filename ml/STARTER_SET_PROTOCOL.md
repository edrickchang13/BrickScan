# BrickScan Starter Set Scan Protocol

A step-by-step guide to building your first real-photo training dataset from a single LEGO set.
The goal is ~30 labeled photos per unique piece with enough lighting and angle variety to
meaningfully close the synthetic-to-real gap.

---

## Recommended Starter Set

**LEGO Classic Large Creative Brick Box — #10698 (or #11717)**

Why this set:
- ~790 pieces, ~33 unique part types — high variety per box
- Every part (2×4 brick, 2×2 plate, 1×2 slope, etc.) appears in thousands of other sets
- ~15 colors present, covering black, white, red, yellow, blue, green, orange
- Total unique (part × color) combinations: ~80 — manageable in a day of shooting

Expected output: ~2,400 photos → after augmentation, ~12K training samples.

---

## Equipment

- **Camera**: iPhone on a tripod or desk stand, or any camera with manual white balance
- **Backgrounds**: one sheet of white A3/poster board + one sheet of mid-grey
- **Lighting**: a desk lamp (warm) + diffuse daylight from a window, OR an LED ring light
- **Turntable**: optional — a lazy susan from the kitchen works perfectly
- **Container**: a small bowl or ring of putty to hold pieces upright

---

## Naming Convention

Each photo filename must encode the part number and color ID so it can be imported automatically:

```
{part_num}_{rebrickable_color_id}_{angle_idx:04d}.jpg

Examples:
  3001_4_0000.jpg   ← 2×4 brick, Red (color 4), angle 0
  3001_4_0001.jpg   ← 2×4 brick, Red, angle 1
  3022_11_0005.jpg  ← 2×2 plate, Black (color 11), angle 5
```

Rebrickable color IDs: https://rebrickable.com/colors/
(The BrickScan app shows the color ID in Part Details → Color ID field)

---

## Shooting Protocol — 30 Photos per Piece

For each unique (part, color) combination, take photos in 3 rounds:

### Round A — Top-Down, White Background (12 shots)
1. Place the white poster board flat on a table
2. Position lamp overhead at ~45° to one side (creates gentle shadows showing texture)
3. Place piece in center, studs facing up
4. Rotate piece 30° between each shot → 12 shots = full 360° coverage
5. Keep camera fixed overhead, parallel to the table surface

### Round B — Side View, Grey Background (12 shots)
1. Switch to grey background
2. Tilt camera 30–45° down from horizontal
3. Same rotation sequence (30° per shot)
4. This captures side stud detail and height/thickness — critical for brick vs plate

### Round C — Natural Light, Random Placement (6 shots)
1. Move near a window with diffuse daylight (no direct sun)
2. No fixed rotation — just varied hand-held angles that feel natural
3. Include 1–2 shots where the piece is slightly out-of-focus or partially shadowed
4. Include 1 shot on a textured surface (e.g. desk wood grain, fabric)

These last 6 shots are the most valuable for real-world generalisation.

---

## Workflow with the BrickScan App

You can use BrickScan itself to verify part numbers as you go:

1. Open BrickScan → Scan tab → Photo mode
2. Scan a piece with the camera to identify it and confirm the part number
3. Check Part Details to get the exact Rebrickable part number and color ID
4. Write that to your tracking sheet, then shoot the 30 photos

---

## Tracking Sheet Template

Keep a CSV file (e.g. `scan_session_log.csv`) as you go:

```csv
part_num,color_id,color_name,shots_taken,notes
3001,4,Red,30,complete
3001,11,Black,30,complete
3022,4,Red,18,in progress
3003,1,Blue,0,not started
```

---

## Importing Photos into the Training Pipeline

Once photos are shot and named correctly, place them in:

```
ml/data/real_scans/
  └── {part_num}_{color_id}/
        ├── 3001_4_0000.jpg
        ├── 3001_4_0001.jpg
        └── ...
```

The dataset loader in `train_contrastive.py` expects `class_dir/image.jpg` where the
class directory name is the class label. Use part-color combined key as the class
to distinguish red 3001 from blue 3001:

```bash
# Reorganize: flatten to part_color class dirs
python ml/data/organize_real_scans.py \
    --input  ml/data/real_scans/ \
    --output ml/data/real_scans_organized/
```

Then pass to training:

```bash
# Fine-tune contrastive model on real scans (after pre-training on renders):
python ml/train_contrastive.py \
    --data-dir ml/data/real_scans_organized/ \
    --output-dir ml/output/real_finetune/ \
    --epochs 20 \
    --lr 5e-5 \
    --batch-size 64
```

---

## Batch Import Helper

The script below can rename photos from a camera roll dump into the correct naming scheme.
Run it after copying photos off your iPhone:

```bash
python ml/data/rename_scans.py \
    --input  ~/Desktop/LEGO_shoot/ \
    --output ml/data/real_scans/ \
    --part-num 3001 \
    --color-id 4 \
    --start-idx 0
```

---

## Quality Checklist

Before moving on from each piece:
- [ ] All 30 photos shot (12 top-down + 12 side + 6 natural)
- [ ] Part number confirmed in BrickScan app
- [ ] Color ID noted in tracking sheet
- [ ] No blurry shots (tap to focus on the piece before shooting)
- [ ] At least 1 shot with piece sideways/inverted (shows underside stud geometry)
- [ ] Photos imported and renamed correctly

---

## Part Priority List for Set #10698

Shoot these first — highest frequency across all LEGO sets, most valuable for training:

| Part | Description          | Colors in set |
|------|----------------------|---------------|
| 3001 | 2×4 Brick            | R, B, Y, W, Bk|
| 3004 | 1×2 Brick            | R, B, Y, W, Bk|
| 3003 | 2×2 Brick            | R, B, Y, G   |
| 3005 | 1×1 Brick            | R, B, Y       |
| 3022 | 2×2 Plate            | R, B, Y, W   |
| 3023 | 1×2 Plate            | R, B, Y, W, Bk|
| 3020 | 2×4 Plate            | R, B, Y, W   |
| 3024 | 1×1 Plate            | Various       |
| 3040 | 2×1 Slope 45°        | R, B, Y       |
| 3039 | 2×3 Slope 45°        | R, B          |
| 3045 | 2×2 Slope 45°        | Various       |
| 3460 | 1×8 Plate            | R, W          |
| 3034 | 2×8 Plate            | R, W          |

---

## Expected Timeline

| Activity | Time |
|---|---|
| Setup (backgrounds, lighting, camera) | 20 min |
| Per piece (30 shots + naming + tracking) | ~4 min |
| 33 unique parts × 80 combinations | ~5–6 hours |
| Import + rename photos | 30 min |
| **Total** | **~7 hours** (spread over 1–2 sessions) |

---

## What This Gives You

After completing this protocol you'll have:
- **~2,400 real labeled photos** across 80 unique (part, color) classes
- A **validation set** (15% holdout = ~360 photos) to measure how your renders → real transfer
- Ground truth to benchmark the current model and measure improvement from LoRA fine-tuning
- A **reusable protocol** — repeat with a second set to expand coverage

The synthetic renders cover the structural/geometric discrimination.
These real photos teach the model texture, lighting variation, and finger-smudge resistance.
