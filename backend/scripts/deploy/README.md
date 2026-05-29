# Flywheel scheduler templates

Timer/cron templates that turn the proven confirm→append flywheel into a running
loop. They drive `backend/scripts/flywheel_refresh_cron.sh`, which calls the
backend's flywheel endpoints over HTTP (same pattern as `weekly_eval_cron.sh`).

Two cadences:

| cadence | what it does | macOS (launchd) | Linux (systemd) |
|---|---|---|---|
| **fast** (every 15 min) | fold confirmed-feedback backlog into the galleries (no retrain) + journal a metrics snapshot | `com.brickscan.flywheel-refresh.plist` | `brickscan-flywheel-refresh.{service,timer}` |
| **heavy** (Mon 03:00) | weekly accuracy snapshot + hand off to `ml/scripts/active_learning_cron.sh` (re-embed / distillation) if it exists | `com.brickscan.flywheel-heavy.plist` | `brickscan-flywheel-heavy.{service,timer}` |

The fast loop alone raises accuracy (the +2pp append result in `ml/FLYWHEEL.md`);
the heavy pass is optional gallery compaction and is a no-op until the Track-D
distillation script lands.

## Install

Replace every `REPLACE_ME` (checkout path, and on Linux the `User=`) first.

**macOS (per-user launchd):**
```bash
cp com.brickscan.flywheel-refresh.plist ~/Library/LaunchAgents/
cp com.brickscan.flywheel-heavy.plist   ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.brickscan.flywheel-refresh.plist
launchctl load ~/Library/LaunchAgents/com.brickscan.flywheel-heavy.plist
launchctl start com.brickscan.flywheel-refresh        # run the fast loop once now
```

**Linux (system-wide systemd):**
```bash
cp brickscan-flywheel-*.{service,timer} /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now brickscan-flywheel-refresh.timer
systemctl enable --now brickscan-flywheel-heavy.timer
systemctl list-timers | grep flywheel
```

**Plain cron** (if you prefer crontab to a timer unit):
```cron
*/15 * * * *  bash /path/to/brickscan/backend/scripts/flywheel_refresh_cron.sh
0 3 * * 1     bash /path/to/brickscan/backend/scripts/flywheel_refresh_cron.sh --heavy
```

## Env vars (read by flywheel_refresh_cron.sh)

| var | default | meaning |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | backend base URL |
| `ADMIN_AUTH_TOKEN` | — | optional bearer token if endpoints require auth |
| `REFRESH_LIMIT` | `1000` | max backlog rows folded per fast pass |
| `ONLY_CORRECTIONS` | `false` | replay only rows where the model was wrong |
| `WINDOW_DAYS` | `30` | accuracy-snapshot window on `--heavy` |
| `HEAVY_SCRIPT` | `<repo>/ml/scripts/active_learning_cron.sh` | heavy-refresh handoff target |
| `FLYWHEEL_LOG` | — | optional path to append a run-log line to |

## Outputs (for monitoring)

- `backend/data/flywheel/refresh_state/last_refresh.json` — last fast-pass summary + cumulative counters
- `backend/data/flywheel/metrics/latest.json` + `<UTC-stamp>.json` — coverage / volume / confusion-pair time series
- `feedback_eval_snapshots` table — weekly top1/top3 accuracy trend (heavy pass)
