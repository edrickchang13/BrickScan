# PR draft — `chore/xcode26-expo-upgrade` → `main`

**Open URL:** https://github.com/edrickchang13/BrickScan/compare/main...chore/xcode26-expo-upgrade

Paste the body below into the PR description.

---

## Title
```
Brickscan v0.6: real-photo classifier, continuous live scan, color head, Brickify-parity sprint
```

## Body
```markdown
## Summary

A long-running branch that grew well beyond its original Xcode/Expo upgrade
scope. Folds in three months of ML, backend, and mobile work into a single
reviewable PR. Every section is independently testable.

## Highlights

| Layer | What's new |
|---|---|
| **ML** | Real-photo MobileNetV3-Large classifier (91.6% xdomain), color classifier (94.5% top-1 / 99.25% top-3 on 60 LEGO colors), YOLOv8-L pile detector (mAP50 87.3%), DINOv2 + LoRA contrastive retrieval pipeline (training in progress) |
| **Backend** | Brickognize → Gemini → local cascade hardened, Grad-CAM explainability endpoint, `/feedback/pending-review` (Mattheij active-learning), `/api/inventory/buildable-sets`, `/api/inventory/analytics`, `/api/inventory/visual-search-status` |
| **Mobile** | **Continuous live-feed scan with per-bbox Kalman tracking** (Phases 1-5), HeatmapExplainer, ReviewQueueScreen, ColorSwatch, ConfirmBricksModal, CollectionAnalyticsScreen, OnboardingScreen 4-slide flow, on-device YOLO via ORT-RN |

## Continuous live scan (the headline feature)

5-phase incremental delivery:

- **Phase 1** — MVP using `/scan/pile`, lock by part_num
- **Phase 2** — switched to `/scan-multi`, per-bbox IoU tracking, live overlay
- **Phase 3** — Kalman bbox smoothing, AsyncStorage session persistence, dev-mode latency telemetry
- **Phase 4** — multi-brick confirmation modal w/ qty edit, drawer sort modes (recent/conf/part/color)
- **Phase 5** — on-device YOLO via `onnxruntime-react-native` + int8 quant, adaptive latency throttle

See `docs/CONTINUOUS_SCAN_PHASE5.md` for stage-by-stage status.

## Real-photo training story

Major mid-PR pivot documented in commit history:

1. Initial synthetic-only training hit 89.3% on synthetic val, 5.7% on real photos
2. Pivoted to real-photo training (Nature 2023 + pacogarciam3 + Hex:Lego crops + Rebrickable CDN)
3. Phase 2 model lands at 91.6% cross-domain top-1
4. ConvNeXt-B teacher + distillation attempted — teacher underperformed, distillation cancelled, no regression shipped
5. Color classifier on Rebrickable CDN canonical photos with `(element_id, color_id)` ground truth → 94.5% top-1

## Brickify-parity sprint (week 1)

| Item | Status |
|---|---|
| Color classifier | ✅ shipped |
| Pricing integration | 🚫 explicitly skipped |
| DINOv2 retrieval | 🔄 model training; backend service + endpoint shipped |
| Theme/year analytics | ✅ shipped (backend + mobile) |
| Set-completion ("what can you build?") | ✅ shipped |
| Onboarding polish | ✅ shipped |
| Brickit benchmark spec | ✅ harness + scoring shipped (manual eval pending) |

## Test plan

- [ ] `cd backend && pytest` — full backend test suite (49 tests + new suites for set_completion, analytics, visual_search)
- [ ] `cd mobile && npm test` — Jest tests including new `bboxTracker.spec.ts`, `kalmanBbox.spec.ts`
- [ ] `cd mobile && npm run lint`
- [ ] `cd mobile && npx tsc --noEmit --skipLibCheck` — type-check
- [ ] Smoke test: scan a brick → response includes part_num + color_name + color_hex
- [ ] Smoke test: continuous scan → bricks lock with green boxes
- [ ] Smoke test: review queue → low-confidence scans surface with editable corrections
- [ ] Smoke test: `/api/inventory/buildable-sets` → returns top sets sorted by completion %

## Rollback plan

Backend models are versioned in git via gitignore allowlist. If anything regresses, revert to `0309207` (the README rebase point) restores everything to the pre-sprint state. Backups also exist locally at `backend/models/backup_pre_*` (not in git but trivially regenerated from prior commits).

## Known limitations

- DINOv2 retrieval catalogue not yet built (training in progress; visual_search service activates automatically when `backend/data/catalog_embeddings.pkl` lands)
- On-device YOLO uses raw 28-class labels; backend classifier round-trip per bbox queued
- Set-scanning POC not started
- Native pixel bridge for on-device inference not started (current JS jpeg-js path is ~50-80ms slower than native)
```
