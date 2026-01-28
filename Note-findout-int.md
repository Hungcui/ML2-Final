# Preprocessing checklist for your own collected digit images
- Standardize foreground/background (background: white, digit: black)
- Remove empty border (crop to the digit)
- Pad to square + resize to 28×28 with anti-aliasing
- 