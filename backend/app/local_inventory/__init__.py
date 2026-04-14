"""
Local Inventory System for BrickScan.

This module handles offline, device-local LEGO brick scanning and inventory management.
Users can scan their physical bricks, build a local database, and handle uncertain predictions.

Includes:
- LocalInventoryPart: Database model for user-scanned parts with confidence tracking
- ScanSession: Tracking of scan sessions (e.g., "Technic 42145")
- Database setup: SQLite at ~/brickscan_inventory.db
- Image handling: Storage and preprocessing for mobile camera input
- Confidence handling: "Known" (>80%) vs "Uncertain" (<80%) predictions
"""
