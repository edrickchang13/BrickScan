# BrickScan vs the field

Honest, technically-grounded comparison against every LEGO ID app worth
considering. Numbers come from public docs, app store descriptions, and
direct testing. Where we don't have hard numbers we say so.

## Apps compared

| App | Platform | Type | Pricing | Closest analogue to BrickScan |
|---|---|---|---|---|
| **BrickScan** (this app) | iOS + Android | Live continuous scan + open ML stack | Free + (planned) Pro tier | — |
| **Brickit** | iOS + Android | Snapshot pile-scan + AR builds | Free / subscription | UX paradigm |
| **Bricksee** | iOS + Android | Photo inventory + multi-piece detection | Freemium | Inventory tracking |
| **Brickify** | iOS + Android | Snapshot pile-scan + price intelligence | Freemium | Collection management |
| **Brickognize** | Web + free API | Single-piece classifier | Free | Backend cascade primary |
| **Instabrick** | Hardware ($200) + web | Camera box + dedicated AI | One-time hardware | None — different category |

## Feature matrix

| Feature | BrickScan | Brickit | Bricksee | Brickify | Brickognize |
|---|:---:|:---:|:---:|:---:|:---:|
| **Live continuous scan** | ✅ Phase 5 (per-bbox Kalman) | ❌ snapshot | ❌ snapshot | ❌ snapshot | ❌ web only |
| Multi-brick detection | ✅ YOLOv8-L (mAP50 87.3%) | ✅ undisclosed | ✅ undisclosed | ✅ undisclosed | ❌ single only |
| Per-bbox tracking across frames | ✅ IoU + Kalman | ❌ | ❌ | ❌ | n/a |
| Color identification | ✅ MobileNetV3-Small (94.5% top-1) | ❌ explicitly ignores color | ✅ undisclosed | ✅ undisclosed | ✅ |
| **Real-photo accuracy (xdomain)** | **91.6% top-1**, 99.4% top-5 | undisclosed | undisclosed | undisclosed | ~85% (paper) |
| Confidence calibration | ✅ per-source temp scaling | ❌ | ❌ | ❌ | ❌ |
| Grad-CAM "why this part?" | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Active learning loop** | ✅ Review Queue + cron retrain | ❌ | partial | claims yes (no mechanism) | ❌ |
| On-device inference | ✅ Phase 5 int8 YOLO (58 MB) | ❌ | ❌ | ❌ | ❌ |
| Offline scan capability | ✅ partial (on-device tier) | ❌ | ❌ | ❌ | ❌ |
| Inventory tracking | ✅ | ❌ | ✅ | ✅ | ❌ |
| Set-completion ("what can you build") | ✅ | ✅ flagship | partial | ❌ | ❌ |
| Theme/year analytics | ✅ | ❌ | ❌ | ✅ | ❌ |
| Real-time pricing | ❌ planned | ❌ | ❌ | ✅ flagship | ❌ |
| Set-level scanning | ❌ POC scaffolded | ❌ | partial | ✅ | ❌ |
| Minifig recognition | ❌ | ❌ | ❌ | ✅ | partial |
| Open ML stack | ✅ | ❌ closed | ❌ closed | ❌ closed | ✅ free API |
| Self-hostable | ✅ docker-compose | ❌ | ❌ | ❌ | ❌ |

## Where BrickScan wins

1. **Continuous live scan UX.** Brickit and Brickify make you take a snapshot and wait. BrickScan keeps the camera live; bricks lock as you sweep. Faster for inventorying a pile of 30+ bricks.
2. **Per-bbox tracking with Kalman smoothing.** Two physically separate 2x4 reds in one frame become two distinct tracks; competitors merge them. Tracks survive momentary occlusion or detector miss.
3. **Real-photo accuracy you can verify.** Our 91.6% xdomain and 94.5% color top-1 are documented and reproducible from the training pipeline in this repo. Closed apps can't be audited.
4. **Active learning loop is mechanism, not marketing.** A weekly cron pulls user corrections from `/feedback/pending-review`, fine-tunes the distillation student, and ships an updated ONNX. Brickify's "more users = smarter" claim has no published mechanism.
5. **Open ML stack.** The whole pipeline (data → training → ONNX → backend → mobile) lives in the repo. You can train your own model on your own bricks if our coverage doesn't fit you.
6. **Grad-CAM explainability.** Tap "why this part?" and see the heatmap. Niche but unique trust signal — none of the closed apps offer this.

## Where BrickScan loses (today)

1. **No pricing layer.** Brickify's killer feature. Implementable in ~3 hours via Rebrickable + BrickLink APIs; deliberately deprioritised.
2. **No set-level scanning.** Brickit recognises whole sets ("hey, that's set #75192 Millennium Falcon"); we'd need a separate model.
3. **Class coverage 1,000 parts.** Brickognize claims 10K+. We bridge the gap with the cascade (Brickognize is our primary tier) so user impact is small, but headline counts look weaker in marketing.
4. **Onboarding polish.** Brickify's first-run flow is genuinely good. Ours is functional but minimal.
5. **No minifig recognition.** Brickify and BrickMonkey both have it. Different model + dataset.

## Where BrickScan and Brickit / Brickify converge but no one wins clearly

- **Multi-piece detection accuracy.** Brickit demos hold up for 30+ bricks in a frame; ours demonstrably matches at the small scales we've tested. Need head-to-head benchmark on a fixed pile.
- **Inventory tracking depth.** All three have inventories. UX preferences vary.
- **App polish.** Brickit and Brickify have years of design iteration on us.

## How we measure what we claim

| Claim | Method | Reproducibility |
|---|---|---|
| 91.6% real-photo top-1 | held-out Hex:Lego val set, xdomain split | `backend/models/export_info.json` + training log |
| 87.3% YOLO mAP50 | Roboflow Hex:Lego v3 val set | `backend/models/yolo_lego.results.csv` |
| 94.5% color top-1 | held-out Rebrickable CDN val (60 colors) | `ml/output/color_v1_*/history.json` |
| Phase 5 latency | iPhone 15 Pro Max, 4 fps continuous | reported in `CONTINUOUS_SCAN_PHASE5.md` |

## How to run a head-to-head yourself

1. Install Brickit + Brickify side by side
2. Lay out 20-30 known bricks (manually noted by part_num + color)
3. Scan with each app under three lighting conditions
4. Score: top-1 accuracy, top-5 accuracy, color accuracy, false positives, time-to-first-result
5. The benchmark harness at `docs/BRICKIT_BENCHMARK.md` walks you through scoring

Or read [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) when populated.
