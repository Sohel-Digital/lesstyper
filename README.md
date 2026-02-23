# lesstyper

lesstyper is a Windows voice typing app with a multi-page desktop UI and global hotkeys. It records your voice, transcribes locally with `faster-whisper`, post-processes text, and inserts it into the focused app.

## Features
- Global hotkeys (works across apps in Windows).
- Full UI pages: `Home`, `History`, `Dictionary`, and `Settings`.
- Visual desktop dashboard with configurable options.
- Floating voice-capture notification (listening/transcribing/inserted).
- Quick start/stop voice capture.
- Fast local transcription (`faster-whisper` on CPU int8).
- Language selector shows English as active; German, French, Spanish, Hindi, and Bengali are marked coming soon (disabled).
- Built-in silence filtering (VAD) plus filler-word cleanup.
- Auto-types text into the active window.
- Persistent dictionary rules saved in `lesstyper_dictionary.json`.
- Bengali font checker is shown only when Bengali is selected.

## Setup
1. Install Python 3.11 or 3.12 (recommended).
2. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m ensurepip --upgrade
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```

## Run
```powershell
python voice_type.py
```

The app opens the `lesstyper` control window with working sidebar pages. While recording, a floating bottom-center capture badge appears.

Default hotkeys:
- `Ctrl+Alt+V`: Start/Stop recording
- `Ctrl+Alt+Q`: Quit

## Useful options
```powershell
python voice_type.py --model base.en --language en --start-stop-hotkey "ctrl+shift+space"
```

Options:
- `--model`: Whisper model (`tiny.en`, `base.en`, `small.en`, etc.)
- `--language`: currently `en` is enabled in UI; other listed languages are marked coming soon.
- `--start-stop-hotkey`: Global toggle key combo
- `--quit-hotkey`: Exit key combo
- `--no-trailing-space`: Do not append space after typed text
- `--keep-fillers`: Keep filler words (`um`, `uh`, etc.)
- `--no-ui`: Run background-only mode (no desktop window)

## Notes
- First run downloads the selected Whisper model.
- Non-English UI language choices are intentionally disabled for now and shown as coming soon.
- In external apps, final text font is controlled by that app/editor (lesstyper inserts Unicode text).
- If global hotkeys fail, run PowerShell as Administrator.
- For lower latency, use `--model tiny.en` or `--model base.en`.
- If you previously used Python 3.13 and saw `numpy`/`html.entities` import errors, use the `py -3.11` environment above.

## macOS DMG via GitHub Actions
- Workflow file: `.github/workflows/macos-dmg.yml`
- Trigger manually from Actions (`workflow_dispatch`) or on pushes to `main` that change app/build files.
- Output artifact name: `lesstyper-macos-dmg` (contains `lesstyper-macOS.dmg`).
- The DMG is unsigned/not notarized by default; add signing/notarization steps before public distribution.
