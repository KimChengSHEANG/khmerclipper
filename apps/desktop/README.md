# KhmerClipper for Windows and Linux

Tray app: select a screen region → Khmer (+English) OCR → text on the clipboard. Python +
PySide6. The macOS app lives in [`../mac`](../mac) and is separate.

## Features

- Region capture: Windows and X11 use a frozen-screenshot overlay. Wayland uses the desktop's own screenshot picker (xdg-desktop-portal).
- OCR engines (Preferences):
  - **TrOCR** (Khmer fine-tune, ONNX): the default. The model (about 670 MB, from Hugging Face) downloads in the background the first time the app opens; capture is available once it finishes.
  - **Tesseract** (`khm+eng`, bundled): works offline with no setup.
- Auto-copy, result popup, history (last 200), OCR Image File…, launch at login, daily update check.
- Global hotkey (default `Ctrl+Shift+2`) on Windows and X11. **Wayland:** add a custom shortcut in your desktop's keyboard settings that runs `khmerclipper --capture` (for the AppImage: `/path/to/KhmerClipper-x.y.z-x86_64.AppImage --capture`).

## Install

Download from [KhmerClipper releases](https://github.com/KimChengSHEANG/khmerclipper/releases):

- **Windows:** `KhmerClipper-<ver>-windows-x64-setup.exe`. Per-user install, no admin. Builds are unsigned, so SmartScreen warns: More info → Run anyway.
- **Linux:** `KhmerClipper-<ver>-x86_64.AppImage`. Run `chmod +x` on it, then start it. GNOME needs the AppIndicator extension for the tray icon (on by default on Ubuntu).

## Develop

```bash
cd apps/desktop
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python tools/fetch_tessdata.py        # pinned khm/eng tessdata
.venv/bin/python -m khmerclipper                   # tray app
.venv/bin/python -m khmerclipper ocr image.png     # headless OCR
.venv/bin/python -m pytest
```

You need a `tesseract` 5.x on `PATH`, or `KHMERCLIPPER_TESSERACT=/path/to/tesseract`. Options: `apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim build on Windows.

## TrOCR models

1. Export: `.venv/bin/pip install -e '.[tools]'`, then `.venv/bin/python tools/export_onnx.py <checkpoint> exported/khmer`. This checks ONNX against torch.
2. Publish (needs `huggingface-cli login`): `.venv/bin/python tools/publish_models.py exported/khmer --version khmer-m2`. This pins the HF commit and the SHA-256 values in `src/khmerclipper/ocr/trocr_manifest.json`. Commit that file.
3. To test a local export without publishing, set `KHMERCLIPPER_TROCR_DIR=exported/khmer`.

## Release

Releases are published to this repo's GitHub Releases with the workflow's own `GITHUB_TOKEN` (no secret to set up). Each release:

1. Bump `__version__` in `src/khmerclipper/__init__.py` and commit.
2. Run `git tag desktop-v<version>` and push the tag (`git push origin desktop-v<version>`).
3. CI tests, builds both installers, smoke-tests the frozen apps, and publishes release `v<version>`. Installed apps see it within a day.
4. CI leaves the release as a **draft** with notes from `docs/releases/v<version>.md`. On a Mac, run `make release-mac` (signed + notarized, see `apps/mac/README.md`), `gh release upload v<version> apps/mac/dist/KhmerClipper-<version>.dmg`, then publish the draft.

Manual checklist before announcing a release:

- [ ] Windows 11 at 100% and 150% scaling, two monitors: capture on each monitor crops exactly the dragged region.
- [ ] Windows: the default `Ctrl+Shift+2` hotkey fires, and a custom hotkey set in Preferences fires after the change.
- [ ] Ubuntu X11: hotkey, capture, popup and clipboard.
- [ ] Ubuntu Wayland: `--capture` shortcut opens the portal picker, and text lands on the clipboard.
- [ ] TrOCR first-use download: progress shown, Cancel works, a retry resumes, and OCR runs after the download.
- [ ] Launch at login toggles and survives a reboot. Uninstall (Windows) removes the Run entry.
- [ ] "Check for Updates…" reports up to date, or offers the newer release.
- [ ] Khmer renders correctly (no boxes) in the popup and in History on a fresh install.
