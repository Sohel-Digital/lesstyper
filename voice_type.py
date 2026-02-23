
import argparse
import ctypes
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import wave
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageTk = None

try:
    import pystray
except Exception:
    pystray = None

try:
    import winreg
except ImportError:
    winreg = None

LANGUAGE_CODE_TO_NAME = {
    "": "Auto detect",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "bn": "Bengali",
}

LANGUAGE_ALIASES = {
    "": "",
    "auto": "",
    "auto detect": "",
    "auto-detect": "",
    "english": "en",
    "en": "en",
    "german": "de",
    "deutsch": "de",
    "de": "de",
    "french": "fr",
    "francais": "fr",
    "fr": "fr",
    "spanish": "es",
    "espanol": "es",
    "es": "es",
    "hindi": "hi",
    "hi": "hi",
    "bengali": "bn",
    "bangla": "bn",
    "bn": "bn",
}

LANGUAGE_DROPDOWN_LABELS = {
    "en": "English (en)",
    "de": "German (de) - Coming soon",
    "fr": "French (fr) - Coming soon",
    "es": "Spanish (es) - Coming soon",
    "hi": "Hindi (hi) - Coming soon",
    "bn": "Bengali (bn) - Coming soon",
}

LANGUAGE_DROPDOWN_VALUES = [LANGUAGE_DROPDOWN_LABELS[code] for code in ["en", "de", "fr", "es", "hi", "bn"]]

ENABLED_LANGUAGE_CODES = {"en"}
DISABLED_LANGUAGE_CODES = ["de", "fr", "es", "hi", "bn"]

MODEL_OPTIONS = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "distil-large-v3",
]

DEFAULT_MULTILINGUAL_MODEL = "small"

BENGALI_FONT_CANDIDATES = [
    "Nirmala UI",
    "SolaimanLipi",
    "Kalpurush",
    "Siyam Rupali",
    "Vrinda",
    "Arial Unicode MS",
]


def normalize_language_input(value: str) -> str:
    raw = value.strip().lower()
    if raw in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[raw]

    match = re.search(r"\(([a-z]{2})\)", raw)
    if match:
        return match.group(1)

    if re.fullmatch(r"[a-z]{2}", raw):
        return raw

    return raw


def language_display_name(code: str) -> str:
    normalized = normalize_language_input(code)
    return LANGUAGE_CODE_TO_NAME.get(normalized, normalized or "Auto detect")


def language_dropdown_value(code: str) -> str:
    normalized = normalize_language_input(code)
    if normalized in LANGUAGE_DROPDOWN_LABELS:
        return LANGUAGE_DROPDOWN_LABELS[normalized]
    if re.fullmatch(r"[a-z]{2}", normalized):
        return f"{normalized.upper()} ({normalized})"
    return LANGUAGE_DROPDOWN_LABELS["en"]


def ensure_model_language_compatibility(model: str, language: str) -> tuple[str, bool]:
    model_clean = model.strip()
    lang = normalize_language_input(language)
    if lang and lang != "en" and model_clean.lower().endswith(".en"):
        return DEFAULT_MULTILINGUAL_MODEL, True
    return model_clean, False


@dataclass
class VoiceTyperConfig:
    start_stop_hotkey: str = "ctrl+alt+v"
    quit_hotkey: str = "ctrl+alt+q"
    sample_rate: int = 16000
    model_size: str = "small.en"
    language: str = "en"
    min_record_seconds: float = 0.4
    trailing_space: bool = True
    remove_fillers: bool = True
    ui_enabled: bool = True


class DictionaryStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> dict[str, str]:
        if not self.file_path.exists():
            return {}
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        cleaned: dict[str, str] = {}
        for key, value in data.items():
            src = str(key).strip()
            dst = str(value).strip()
            if src and dst:
                cleaned[src] = dst
        return cleaned

    def save(self, entries: dict[str, str]) -> None:
        self.file_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


class VoiceTyperApp:
    APP_NAME = "lesstyper"
    STARTUP_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
    STARTUP_VALUE_NAME = APP_NAME
    LEGACY_STARTUP_VALUE_NAMES = ["NoType.app", "typenada"]
    DICTIONARY_FILE = Path("lesstyper_dictionary.json")
    LEGACY_DICTIONARY_FILES = [Path("notype_dictionary.json"), Path("typenada_dictionary.json")]

    def __init__(self, config: VoiceTyperConfig) -> None:
        self.config = config
        self.config.language = normalize_language_input(self.config.language)
        if self.config.language not in ENABLED_LANGUAGE_CODES:
            print(
                f"{language_display_name(self.config.language)} is coming soon. Falling back to English."
            )
            self.config.language = "en"
        compatible_model, switched = ensure_model_language_compatibility(
            self.config.model_size,
            self.config.language,
        )
        self.config.model_size = compatible_model
        self.model_lock = threading.Lock()
        self.model = WhisperModel(self.config.model_size, device="cpu", compute_type="int8")

        self.recording = False
        self.stream: sd.InputStream | None = None
        self.frames: list[np.ndarray] = []
        self.frames_lock = threading.Lock()

        self.shutdown_event = threading.Event()
        self.jobs: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._process_loop, daemon=True)

        self.hotkey_handles: list[int] = []
        self.events: queue.Queue[dict] | None = None
        self.last_level_emit = 0.0

        self.total_seconds = 0.0
        self.total_words = 0
        self.total_clips = 0
        self.history_records: list[tuple[str, str]] = []

        self.model_reload_lock = threading.Lock()
        self.model_reloading = False

        self.target_window_handle = 0

        self.dictionary_store = DictionaryStore(self._resolve_dictionary_path())
        self.dictionary_entries = self.dictionary_store.load()
        self.dictionary_patterns: list[tuple[list[re.Pattern], str]] = []
        self._refresh_dictionary_patterns()

        if switched:
            print(
                f"Non-English language selected. Switched to multilingual model '{self.config.model_size}'."
            )

    @classmethod
    def _resolve_dictionary_path(cls) -> Path:
        if cls.DICTIONARY_FILE.exists():
            return cls.DICTIONARY_FILE

        for legacy_path in cls.LEGACY_DICTIONARY_FILES:
            if not legacy_path.exists():
                continue
            try:
                legacy_path.replace(cls.DICTIONARY_FILE)
                return cls.DICTIONARY_FILE
            except Exception:
                try:
                    cls.DICTIONARY_FILE.write_bytes(legacy_path.read_bytes())
                    return cls.DICTIONARY_FILE
                except Exception:
                    return legacy_path

        return cls.DICTIONARY_FILE

    def _startup_command(self) -> str:
        if getattr(sys, "frozen", False):
            parts = [sys.executable, *sys.argv[1:]]
        else:
            parts = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        return subprocess.list2cmdline(parts)

    def is_start_with_windows_enabled(self) -> bool:
        if winreg is None:
            return False
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.STARTUP_REG_PATH,
                0,
                winreg.KEY_READ,
            ) as key:
                startup_names = [self.STARTUP_VALUE_NAME, *self.LEGACY_STARTUP_VALUE_NAMES]
                for value_name in startup_names:
                    try:
                        value, _ = winreg.QueryValueEx(key, value_name)
                    except FileNotFoundError:
                        continue
                    if str(value).strip():
                        return True
            return False
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def set_start_with_windows(self, enabled: bool) -> tuple[bool, str]:
        if winreg is None:
            return False, "Startup registration is only available on Windows."

        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, self.STARTUP_REG_PATH) as key:
                if enabled:
                    winreg.SetValueEx(
                        key,
                        self.STARTUP_VALUE_NAME,
                        0,
                        winreg.REG_SZ,
                        self._startup_command(),
                    )
                    for legacy_name in self.LEGACY_STARTUP_VALUE_NAMES:
                        try:
                            winreg.DeleteValue(key, legacy_name)
                        except FileNotFoundError:
                            pass
                    return True, "Start with Windows enabled."

                for value_name in [self.STARTUP_VALUE_NAME, *self.LEGACY_STARTUP_VALUE_NAMES]:
                    try:
                        winreg.DeleteValue(key, value_name)
                    except FileNotFoundError:
                        pass
                return True, "Start with Windows disabled."
        except OSError as exc:
            return False, f"Could not update startup setting: {exc}"

    def attach_ui_queue(self, events: queue.Queue[dict]) -> None:
        self.events = events

    def start(self) -> None:
        if not self.worker.is_alive():
            self.worker.start()
        self._bind_hotkeys()
        self._emit("status", value="idle")
        self._emit_stats()
        self._emit("dictionary_updated", entries=self.get_dictionary_entries())
        self._emit("message", text=f"{self.APP_NAME} is ready")

    def run_headless(self) -> None:
        self.start()
        print(f"{self.APP_NAME} is running.")
        print(f"Start/Stop recording: {self.config.start_stop_hotkey}")
        print(f"Quit: {self.config.quit_hotkey}")
        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.15)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        self._stop_recording_stream()
        self._unbind_hotkeys()
        self._emit("status", value="stopped")
        self._emit("quit")

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def stop_recording(self) -> None:
        if self.recording:
            self._stop_recording()

    def clear_history(self) -> None:
        self.history_records = []
        self._emit("history_cleared")

    def get_dictionary_entries(self) -> list[tuple[str, str]]:
        items = list(self.dictionary_entries.items())
        items.sort(key=lambda item: item[0].lower())
        return items

    def set_dictionary_entries(self, entries: list[tuple[str, str]]) -> tuple[bool, str]:
        cleaned: dict[str, str] = {}
        for src, dst in entries:
            src_clean = str(src).strip()
            dst_clean = str(dst).strip()
            if src_clean and dst_clean:
                cleaned[src_clean] = dst_clean

        self.dictionary_entries = cleaned
        self._refresh_dictionary_patterns()
        try:
            self.dictionary_store.save(cleaned)
        except Exception as exc:
            return False, f"Could not save dictionary: {exc}"

        self._emit("dictionary_updated", entries=self.get_dictionary_entries())
        return True, "Dictionary saved"

    def apply_settings(
        self,
        start_hotkey: str,
        quit_hotkey: str,
        language: str,
        min_seconds: float,
        trailing_space: bool,
        remove_fillers: bool,
        model_size: str,
    ) -> tuple[bool, str]:
        start_hotkey = start_hotkey.strip().lower()
        quit_hotkey = quit_hotkey.strip().lower()
        language = normalize_language_input(language)
        model_size = model_size.strip()

        if not start_hotkey or not quit_hotkey:
            return False, "Hotkeys cannot be empty."
        if start_hotkey == quit_hotkey:
            return False, "Start/Stop and Quit hotkeys must be different."
        if language and not re.fullmatch(r"[a-z]{2}", language):
            return False, "Language must be auto detect or a two-letter code."
        if language not in ENABLED_LANGUAGE_CODES:
            return (
                False,
                f"{language_display_name(language)} support is coming soon. English is available now.",
            )
        if min_seconds < 0.2:
            return False, "Min seconds must be >= 0.2."

        selected_model = model_size or self.config.model_size
        compatible_model, auto_switched = ensure_model_language_compatibility(
            selected_model,
            language,
        )

        self.config.start_stop_hotkey = start_hotkey
        self.config.quit_hotkey = quit_hotkey
        self.config.language = language
        self.config.min_record_seconds = min_seconds
        self.config.trailing_space = trailing_space
        self.config.remove_fillers = remove_fillers

        self._bind_hotkeys()

        should_reload_model = compatible_model != self.config.model_size
        if should_reload_model:
            self._reload_model_async(compatible_model)
            if auto_switched:
                return (
                    True,
                    f"Settings saved. Switched to multilingual model '{compatible_model}' for {language_display_name(language)}. Loading model...",
                )
            return True, "Settings saved. Loading model in background..."

        if auto_switched:
            return (
                True,
                f"Settings saved. Using multilingual model '{compatible_model}' for {language_display_name(language)}.",
            )

        return True, "Settings saved."

    def _reload_model_async(self, model_size: str) -> None:
        with self.model_reload_lock:
            if self.model_reloading:
                self._emit("message", text="Model load already in progress")
                return
            self.model_reloading = True

        def _load() -> None:
            self._emit("message", text=f"Loading model: {model_size}")
            try:
                new_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                with self.model_lock:
                    self.model = new_model
                self.config.model_size = model_size
                self._emit("message", text=f"Model ready: {model_size}")
            except Exception as exc:
                self._emit("error", text=f"Model load failed: {exc}")
            finally:
                with self.model_reload_lock:
                    self.model_reloading = False

        threading.Thread(target=_load, daemon=True).start()

    def _bind_hotkeys(self) -> None:
        self._unbind_hotkeys()
        try:
            self.hotkey_handles.append(
                keyboard.add_hotkey(self.config.start_stop_hotkey, self.toggle_recording)
            )
            self.hotkey_handles.append(
                keyboard.add_hotkey(self.config.quit_hotkey, self.stop)
            )
            self._emit(
                "hotkeys",
                start=self.config.start_stop_hotkey,
                quit=self.config.quit_hotkey,
            )
        except Exception as exc:
            self._emit("error", text=f"Hotkey registration failed: {exc}")

    def _unbind_hotkeys(self) -> None:
        for handle in self.hotkey_handles:
            try:
                keyboard.remove_hotkey(handle)
            except Exception:
                pass
        self.hotkey_handles = []

    def _start_recording(self) -> None:
        if self.recording or self.shutdown_event.is_set():
            return

        self.target_window_handle = self._get_foreground_window()
        self.recording = True
        with self.frames_lock:
            self.frames = []

        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self._emit("status", value="listening")
            self._emit("message", text="Listening...")
        except Exception as exc:
            self.recording = False
            self._emit("error", text=f"Microphone error: {exc}")

    def _stop_recording(self) -> None:
        if not self.recording:
            return

        self.recording = False
        self._stop_recording_stream()

        with self.frames_lock:
            if not self.frames:
                self._emit("status", value="idle")
                self._emit("message", text="No audio captured")
                return
            audio = np.concatenate(self.frames, axis=0).squeeze()
            self.frames = []

        duration_seconds = len(audio) / float(self.config.sample_rate)
        if duration_seconds < self.config.min_record_seconds:
            self._emit("status", value="idle")
            self._emit("message", text="Recording too short")
            return

        self.total_seconds += duration_seconds
        self.total_clips += 1
        self._emit_stats()

        if self.jobs.full():
            try:
                _ = self.jobs.get_nowait()
                self.jobs.task_done()
            except queue.Empty:
                pass

        self.jobs.put(audio)
        self._emit("status", value="processing")
        self._emit("message", text="Transcribing...")

    def _stop_recording_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._emit("error", text=f"Audio status: {status}")

        with self.frames_lock:
            self.frames.append(indata.copy())

        now = time.monotonic()
        if now - self.last_level_emit >= 0.05:
            self.last_level_emit = now
            rms = float(np.sqrt(np.mean(np.square(indata))))
            self._emit("level", value=min(rms * 8.0, 1.0))

    def _process_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                audio = self.jobs.get(timeout=0.15)
            except queue.Empty:
                continue

            try:
                text = self._transcribe(audio)
                if text:
                    inserted = self._inject_text(text)
                    words = len(re.findall(r"\b[\w']+\b", text))
                    self.total_words += words
                    self._append_history(text)
                    self._emit("text", value=text)
                    if inserted:
                        self._emit("message", text="Inserted in active app")
                    else:
                        self._emit("message", text="Transcribed, but insert may have failed")
                else:
                    self._emit("message", text="No speech detected")

                self._emit_stats(last_text=text)
                self._emit("status", value="idle")
            except Exception as exc:
                self._emit("error", text=f"Transcription failed: {exc}")
                self._emit("status", value="idle")
            finally:
                self.jobs.task_done()

    def _transcribe(self, audio: np.ndarray) -> str:
        wav_path = self._write_temp_wav(audio)
        try:
            with self.model_lock:
                model = self.model
            initial_prompt = None
            if self.config.language == "bn":
                initial_prompt = (
                    "বাংলা ভাষার অডিওকে শুধু বাংলা লিপিতে লিখুন। "
                    "ইংরেজি অক্ষরে ট্রান্সলিটারেশন করবেন না।"
                )
            segments, _ = model.transcribe(
                wav_path,
                language=self.config.language or None,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return self._cleanup_text(text)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        clipped = np.clip(audio, -1.0, 1.0)
        int16_audio = (clipped * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(int16_audio.tobytes())

        return wav_path

    def _cleanup_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        if self.config.remove_fillers:
            text = re.sub(
                r"\b(?:um+|uh+|erm|hmm|like)\b",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = self._apply_dictionary_rules(text)
        if text and re.match(r"[A-Za-z]", text[0]):
            text = text[0].upper() + text[1:]
        return text

    def _apply_dictionary_rules(self, text: str) -> str:
        if not self.dictionary_patterns:
            return text

        output = text
        for patterns, replacement in self.dictionary_patterns:
            for pattern in patterns:
                output = pattern.sub(replacement, output)
        return output

    def _refresh_dictionary_patterns(self) -> None:
        compiled: list[tuple[list[re.Pattern], str]] = []
        items = sorted(
            self.dictionary_entries.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for source, replacement in items:
            normalized = " ".join(source.strip().split())
            if not normalized:
                continue

            tokens = [part for part in re.split(r"\s+", normalized) if part]
            raw_patterns: list[str] = []
            seen: set[str] = set()

            def add_pattern(pattern_text: str) -> None:
                if pattern_text and pattern_text not in seen:
                    seen.add(pattern_text)
                    raw_patterns.append(pattern_text)

            if tokens:
                escaped = [re.escape(token) for token in tokens]
                if len(escaped) == 1:
                    add_pattern(escaped[0])
                else:
                    add_pattern(r"\s+".join(escaped))
                    add_pattern(r"(?:[\s._-]+)".join(escaped))
                    add_pattern(r"(?:[\s._-]*)".join(escaped))
                    add_pattern("".join(escaped))
            else:
                add_pattern(re.escape(normalized))

            prefix = r"(?<!\w)" if normalized[0].isalnum() else ""
            suffix = r"(?!\w)" if normalized[-1].isalnum() else ""

            entry_patterns = [
                re.compile(prefix + pattern_text + suffix, flags=re.IGNORECASE)
                for pattern_text in raw_patterns
            ]
            compiled.append((entry_patterns, replacement))

        self.dictionary_patterns = compiled

    def _inject_text(self, text: str) -> bool:
        output = text + (" " if self.config.trailing_space else "")
        self._release_modifier_keys()
        if self.target_window_handle:
            self._focus_window(self.target_window_handle)
            time.sleep(0.06)

        try:
            keyboard.write(output, delay=0)
            return True
        except Exception:
            previous_clipboard: str | None = None
            try:
                previous_clipboard = pyperclip.paste()
            except Exception:
                previous_clipboard = None

            try:
                pyperclip.copy(output)
                keyboard.send("ctrl+v")
                if previous_clipboard is not None:
                    time.sleep(0.05)
                    pyperclip.copy(previous_clipboard)
                return True
            except Exception:
                return False

    def _release_modifier_keys(self) -> None:
        keys = [
            "ctrl",
            "alt",
            "shift",
            "left ctrl",
            "right ctrl",
            "left alt",
            "right alt",
            "left shift",
            "right shift",
        ]
        for key_name in keys:
            try:
                keyboard.release(key_name)
            except Exception:
                pass

    def _get_foreground_window(self) -> int:
        try:
            return int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            return 0

    def _focus_window(self, hwnd: int) -> None:
        if hwnd <= 0:
            return
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _append_history(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.history_records.append((ts, text))
        if len(self.history_records) > 500:
            self.history_records = self.history_records[-500:]
        self._emit("history_item", timestamp=ts, text=text)

    def _emit_stats(self, last_text: str = "") -> None:
        wpm = 0.0
        if self.total_seconds > 0:
            wpm = self.total_words / (self.total_seconds / 60.0)
        self._emit(
            "stats",
            words=self.total_words,
            seconds=self.total_seconds,
            clips=self.total_clips,
            wpm=wpm,
            last_text=last_text,
        )

    def _emit(self, event_type: str, **payload) -> None:
        if self.events is None:
            if event_type in {"message", "error"}:
                print(payload.get("text", ""))
            return
        try:
            self.events.put_nowait({"type": event_type, **payload})
        except queue.Full:
            pass


class CaptureOverlay:
    def __init__(self, root: tk.Tk, on_stop=None) -> None:
        self.root = root
        self.on_stop = on_stop
        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)

        self.pill_bg = "#141a2a"
        self.pill_bg_top = "#1b2340"
        self.pill_border = "#eaf0ff"
        self.shadow_color = "#04070f"
        self.bar_color = "#ffffff"

        self.width = 194
        self.height = 60
        # Use a unique color key that is never used in visible pixels.
        self.transparent_key = "#ff00ff"
        self.transparent_supported = False
        try:
            self.win.configure(bg=self.transparent_key)
            self.win.wm_attributes("-transparentcolor", self.transparent_key)
            self.transparent_supported = True
        except Exception:
            self.win.configure(bg=self.pill_bg)

        self.canvas = tk.Canvas(
            self.win,
            width=self.width,
            height=self.height,
            bg=self.transparent_key if self.transparent_supported else self.pill_bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", lambda _e: self._on_stop_clicked())

        self.wave_left = 0.0
        self.wave_right = 0.0
        self._draw_pill_background()

        self.bars: list[int] = []
        self.bar_profile = [0.36, 0.62, 0.86, 1.0, 0.86, 0.62, 0.36]
        self.base_h = 6.0
        self.max_h = 26.0
        self.bar_heights = [self.base_h for _ in range(len(self.bar_profile))]
        self.bar_velocity = [0.0 for _ in range(len(self.bar_profile))]
        self.bar_targets = [self.base_h for _ in range(len(self.bar_profile))]
        self._build_wave_bars()

        self.state = "idle"
        self.level = 0.0
        self.visual_level = 0.0
        self.voice_energy = 0.0
        self.tick = 0.0
        self.hide_job = None
        self.anim_job = None

    def _draw_pill_background(self) -> None:
        self.canvas.delete("pill")

        # Competitor-style: compact capsule, thin bright border, dark navy body.
        cy = self.height / 2.0
        shadow_w = self.height - 8
        outer_w = self.height - 16
        fill_w = outer_w - 2
        line_margin = 12.0
        left = line_margin + (outer_w / 2.0)
        right = self.width - line_margin - (outer_w / 2.0)

        self.canvas.create_line(
            left,
            cy + 2.3,
            right,
            cy + 2.3,
            fill=self.shadow_color,
            width=shadow_w,
            capstyle=tk.ROUND,
            tags="pill",
        )
        self.canvas.create_line(
            left,
            cy + 0.6,
            right,
            cy + 0.6,
            fill="#0a1020",
            width=shadow_w - 4,
            capstyle=tk.ROUND,
            tags="pill",
        )
        self.canvas.create_line(
            left,
            cy,
            right,
            cy,
            fill=self.pill_border,
            width=outer_w,
            capstyle=tk.ROUND,
            tags="pill",
        )
        self.canvas.create_line(
            left,
            cy,
            right,
            cy,
            fill=self.pill_bg,
            width=fill_w,
            capstyle=tk.ROUND,
            tags="pill",
        )
        self.canvas.create_line(
            left,
            cy - (fill_w / 2.0) + 2.4,
            right,
            cy - (fill_w / 2.0) + 2.4,
            fill=self.pill_bg_top,
            width=1,
            capstyle=tk.ROUND,
            tags="pill",
        )

        center_pad = 66.0
        self.wave_left = (self.width / 2.0) - center_pad / 2.0
        self.wave_right = (self.width / 2.0) + center_pad / 2.0

    def _build_wave_bars(self) -> None:
        for bar in self.bars:
            self.canvas.delete(bar)
        self.bars.clear()

        count = len(self.bar_profile)
        span = max(16.0, self.wave_right - self.wave_left)
        gap = span / max(1, count - 1)
        bar_width = 3.0
        center_y = self.height / 2.0
        for idx in range(count):
            x = self.wave_left + (idx * gap)
            bar = self.canvas.create_line(
                x,
                center_y - self.base_h / 2.0,
                x,
                center_y + self.base_h / 2.0,
                fill=self.bar_color,
                width=bar_width,
                capstyle=tk.ROUND,
            )
            self.bars.append(bar)

    def show_listening(self) -> None:
        self._cancel_hide()
        self.state = "listening"
        self._show()
        self._start_animation()

    def show_processing(self) -> None:
        self._cancel_hide()
        self.state = "processing"
        self._show()
        self._start_animation()

    def show_inserted(self, text: str) -> None:
        self.state = "inserted"
        self.set_level(0.0)
        self._show()
        self._start_animation()
        self.hide_job = self.root.after(1400, self.hide)

    def set_level(self, value: float) -> None:
        self.level = max(0.0, min(1.0, value))

    def hide(self) -> None:
        self._cancel_hide()
        if self.anim_job is not None:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.state = "idle"
        self.level = 0.0
        self.visual_level = 0.0
        self.voice_energy = 0.0
        self.tick = 0.0
        self.bar_heights = [self.base_h for _ in self.bar_heights]
        self.bar_velocity = [0.0 for _ in self.bar_velocity]
        self.bar_targets = [self.base_h for _ in self.bar_targets]
        self.win.withdraw()

    def destroy(self) -> None:
        self._cancel_hide()
        if self.anim_job is not None:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.win.destroy()

    def _show(self) -> None:
        self._place_bottom_center()
        self.win.deiconify()
        self.win.lift()

    def _place_bottom_center(self) -> None:
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - self.width) // 2
        y = sh - self.height - 88
        self.win.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def _cancel_hide(self) -> None:
        if self.hide_job is not None:
            self.root.after_cancel(self.hide_job)
            self.hide_job = None

    def _start_animation(self) -> None:
        if self.anim_job is None:
            self.anim_job = self.root.after(0, self._tick_animation)

    def _tick_animation(self) -> None:
        self.anim_job = None
        if self.state == "idle" or not self.win.winfo_viewable():
            return

        self.tick += 1.0
        blend = 0.20 if self.level > self.visual_level else 0.08
        self.visual_level += (self.level - self.visual_level) * blend
        level_target = max(0.0, min(1.0, (self.visual_level - 0.012) / 0.82))
        self.voice_energy += (level_target - self.voice_energy) * 0.16

        center_y = self.height / 2.0

        for idx, bar in enumerate(self.bars):
            phase = (self.tick * 0.14) - (idx * 0.55)
            if self.state == "listening":
                travel = 0.5 + (0.5 * math.sin(phase))
                amp = (0.09 * self.bar_profile[idx]) + (
                    self.voice_energy * (0.26 + (0.72 * travel)) * self.bar_profile[idx]
                )
            elif self.state == "processing":
                drift = 0.5 + (0.5 * math.sin((self.tick * 0.10) - (idx * 0.72)))
                amp = (0.12 * self.bar_profile[idx]) + (0.12 * drift * self.bar_profile[idx])
            else:
                drift = 0.5 + (0.5 * math.sin((self.tick * 0.08) - (idx * 0.60)))
                amp = (0.08 * self.bar_profile[idx]) + (0.08 * drift * self.bar_profile[idx])

            target_h = self.base_h + (amp * (self.max_h - self.base_h))
            self.bar_targets[idx] = target_h

            stiffness = 0.13 if self.state == "listening" else 0.10
            damping = 0.89 if self.state == "listening" else 0.92
            delta = self.bar_targets[idx] - self.bar_heights[idx]
            self.bar_velocity[idx] = (self.bar_velocity[idx] * damping) + (delta * stiffness)
            self.bar_heights[idx] += self.bar_velocity[idx]
            self.bar_heights[idx] = max(self.base_h, min(self.max_h, self.bar_heights[idx]))

            h = self.bar_heights[idx]
            x1, _, x2, _ = self.canvas.coords(bar)
            self.canvas.coords(bar, x1, center_y - (h / 2.0), x2, center_y + (h / 2.0))
            self.canvas.itemconfigure(bar, fill=self.bar_color)

        self.anim_job = self.root.after(16, self._tick_animation)

    def _on_stop_clicked(self) -> None:
        if callable(self.on_stop):
            try:
                self.on_stop()
            except Exception:
                pass

class VoiceTyperUI:
    def __init__(self, app: VoiceTyperApp) -> None:
        self.app = app
        self.events: queue.Queue[dict] = queue.Queue(maxsize=250)
        self.app.attach_ui_queue(self.events)

        self.c_bg = "#0b141a"
        self.c_surface = "#111b21"
        self.c_sidebar = "#202c33"
        self.c_border = "#2a3942"
        self.c_text = "#e9edef"
        self.c_muted = "#8696a0"
        self.c_primary = "#25d366"
        self.c_primary_hover = "#1faa52"
        self.c_success = "#25d366"
        self.c_success_hover = "#1faa52"
        self.c_danger = "#f15c6d"
        self.c_danger_hover = "#dc4d5f"
        self.c_soft = "#233138"
        self.c_dark = "#0f171c"

        self.root = tk.Tk()
        self.root.title(self.app.APP_NAME)
        self.root.geometry("1220x780")
        self.root.minsize(1020, 700)
        self.root.configure(bg=self.c_bg)
        self.icon_font_path = self._find_fontawesome_font_path()
        self.action_icon_cache: dict[tuple, tk.PhotoImage] = {}
        self.app_icon = self._create_recorder_icon(size=64, bg_color=self.c_bg)
        self.brand_icon = self._create_recorder_icon(size=20, bg_color=self.c_sidebar)
        try:
            self.root.iconphoto(True, self.app_icon)
        except Exception:
            pass

        self.default_text_font = ("Segoe UI", 11)
        self.default_mono_font = ("Consolas", 10)
        self.default_tree_font = ("Segoe UI", 10)
        self.default_tree_heading_font = ("Segoe UI", 10, "bold")
        self.available_bengali_fonts = self._find_available_bengali_fonts()
        self.selected_bengali_font = (
            self.available_bengali_fonts[0] if self.available_bengali_fonts else "Segoe UI"
        )
        self.bengali_text_font = (self.selected_bengali_font, 12)
        self.bengali_mono_font = (self.selected_bengali_font, 12)
        self.bengali_tree_font = (self.selected_bengali_font, 11)

        self.overlay = CaptureOverlay(self.root, on_stop=self._stop_recording_from_overlay)
        self.closing = False

        self.status_var = tk.StringVar(value="Ready")
        self.bengali_font_status_var = tk.StringVar(value="")
        self.hotkey_hint_var = tk.StringVar(value="")
        self.words_var = tk.StringVar(value="0")
        self.seconds_var = tk.StringVar(value="0.0 min")
        self.time_saved_var = tk.StringVar(value="0.0 min")
        self.wpm_var = tk.StringVar(value="0")
        self.last_text_var = tk.StringVar(value="-")

        self.start_hotkey_var = tk.StringVar(value=self.app.config.start_stop_hotkey)
        self.quit_hotkey_var = tk.StringVar(value=self.app.config.quit_hotkey)
        self.language_var = tk.StringVar(
            value=language_dropdown_value(self.app.config.language)
        )
        self.model_var = tk.StringVar(value=self.app.config.model_size)
        self.min_seconds_var = tk.StringVar(value=str(self.app.config.min_record_seconds))
        self.trailing_space_var = tk.BooleanVar(value=self.app.config.trailing_space)
        self.remove_fillers_var = tk.BooleanVar(value=self.app.config.remove_fillers)
        self.start_with_windows_var = tk.BooleanVar(
            value=self.app.is_start_with_windows_enabled()
        )

        self.dictionary_source_var = tk.StringVar(value="")
        self.dictionary_target_var = tk.StringVar(value="")

        self.page_frames: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.nav_button_icons: dict[str, tuple[tk.PhotoImage | None, tk.PhotoImage | None]] = {}
        self.rail_buttons: dict[str, tk.Button] = {}
        self.rail_button_icons: dict[str, tuple[tk.PhotoImage | None, tk.PhotoImage | None]] = {}
        self.page_title_var = tk.StringVar(value="Home")

        self.home_toggle_btn: tk.Button | None = None
        self.home_toggle_caption: tk.Label | None = None
        self.home_start_canvas: tk.Canvas | None = None
        self.home_title_label: tk.Label | None = None
        self.settings_toggle_btn: tk.Button | None = None
        self.history_text: tk.Text | None = None
        self.dictionary_tree: ttk.Treeview | None = None
        self.dictionary_source_entry: tk.Entry | None = None
        self.dictionary_target_entry: tk.Entry | None = None
        self.latest_text_label: tk.Label | None = None
        self.bengali_font_status_label: tk.Label | None = None
        self.tree_style = ttk.Style(self.root)
        self.auto_minimized_on_record = False
        self.tray_icon = None
        self.tray_thread: threading.Thread | None = None
        self.close_to_tray = True
        self.custom_titlebar_enabled = os.name == "nt"
        self.titlebar_max_btn: tk.Button | None = None
        self.window_is_maximized = False
        self.window_restore_geometry = ""
        self._drag_start_x = 0
        self._drag_start_y = 0
        if self.custom_titlebar_enabled:
            try:
                self.root.overrideredirect(True)
            except Exception:
                self.custom_titlebar_enabled = False
        self._configure_ttk_theme()

        self._build_layout()
        self._apply_language_fonts()
        self._update_bengali_font_status(push_status=True)
        self._refresh_hotkeys_hint()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.custom_titlebar_enabled:
            self.root.bind("<Map>", self._on_root_map)
            self.root.bind("<Alt-F4>", self._on_alt_f4)

    def run(self) -> None:
        self.app.start()
        self._poll_events()
        self.root.mainloop()

    def _configure_ttk_theme(self) -> None:
        try:
            self.tree_style.theme_use("clam")
        except Exception:
            pass

        self.tree_style.configure(
            "LessTyper.TCombobox",
            fieldbackground=self.c_surface,
            background="#112430",
            foreground=self.c_text,
            bordercolor=self.c_border,
            lightcolor=self.c_border,
            darkcolor=self.c_border,
            arrowcolor=self.c_text,
            insertcolor=self.c_text,
            arrowsize=15,
            padding=2,
        )
        self.tree_style.map(
            "LessTyper.TCombobox",
            fieldbackground=[("readonly", self.c_surface), ("active", self.c_surface)],
            background=[("readonly", "#112430"), ("active", "#173443")],
            bordercolor=[("focus", self.c_primary), ("readonly", self.c_border)],
            lightcolor=[("focus", self.c_primary), ("readonly", self.c_border)],
            darkcolor=[("focus", self.c_primary), ("readonly", self.c_border)],
            arrowcolor=[("readonly", self.c_text), ("active", "#ffffff")],
            selectbackground=[("readonly", self.c_soft)],
            selectforeground=[("readonly", self.c_text)],
            foreground=[("readonly", self.c_text), ("active", self.c_text)],
        )
        self.tree_style.configure(
            "LessTyper.Treeview",
            background=self.c_surface,
            fieldbackground=self.c_surface,
            foreground=self.c_text,
            bordercolor=self.c_border,
            rowheight=28,
        )
        self.tree_style.map(
            "LessTyper.Treeview",
            background=[("selected", self.c_soft)],
            foreground=[("selected", self.c_text)],
        )
        self.tree_style.configure(
            "LessTyper.Treeview.Heading",
            background=self.c_soft,
            foreground=self.c_text,
            bordercolor=self.c_border,
            font=self.default_tree_heading_font,
        )

    def _create_recorder_icon(
        self,
        size: int = 64,
        bg_color: str = "#ffffff",
    ) -> tk.PhotoImage:
        fa_brand_icon = self._build_fa_brand_icon(size=size, bg_color=bg_color)
        if fa_brand_icon is not None:
            return fa_brand_icon

        img = tk.PhotoImage(width=size, height=size)
        img.put(bg_color, to=(0, 0, size, size))

        # Rounded badge background for both app icon and sidebar logo.
        mid = size / 2.0
        outer_r = max(5, int(size * 0.49))
        inner_r = max(4, int(size * 0.42))
        glow_r = max(2, int(size * 0.2))
        self._draw_aa_circle(img, mid, mid, outer_r, "#1d4ed8", bg_color)
        self._draw_aa_circle(img, mid, mid, inner_r, "#2563eb", bg_color)
        self._draw_aa_circle(
            img,
            max(1.0, mid - (size * 0.14)),
            max(1.0, mid - (size * 0.14)),
            glow_r,
            "#60a5fa",
            "#2563eb",
        )

        dark = "#ffffff"
        accent = "#ef4444"

        # Mic capsule
        capsule_w = max(6, int(size * 0.32))
        x1 = max(1, int(mid - (capsule_w / 2)))
        x2 = min(size - 1, int(mid + (capsule_w / 2)))
        y1 = max(2, int(size * 0.12))
        y2 = max(y1 + 4, int(size * 0.58))
        img.put(dark, to=(x1, y1 + capsule_w // 2, x2, y2 - capsule_w // 2))
        self._draw_aa_circle(img, mid, y1 + (capsule_w / 2), capsule_w / 2, dark, "#2563eb")
        self._draw_aa_circle(img, mid, y2 - (capsule_w / 2), capsule_w / 2, dark, "#2563eb")

        # Stem and base
        stem_w = max(2, int(size * 0.08))
        stem_h = max(3, int(size * 0.16))
        stem_y2 = min(size - 3, y2 + stem_h)
        img.put(
            dark,
            to=(int(mid - stem_w // 2), y2, int(mid + stem_w // 2), stem_y2),
        )
        base_w = max(8, int(size * 0.46))
        base_h = max(2, int(size * 0.05))
        bx1 = int(mid - base_w // 2)
        bx2 = int(mid + base_w // 2)
        by1 = min(size - base_h - 1, stem_y2)
        img.put("#dbeafe", to=(bx1, by1, bx2, by1 + base_h))

        # Record indicator dot
        dot_r = max(2, int(size * 0.11))
        dot_cx = size - dot_r - 3
        dot_cy = size - dot_r - 3
        self._draw_aa_circle(
            img,
            dot_cx,
            dot_cy,
            dot_r + max(1, size // 36),
            "#ffffff",
            "#2563eb",
        )
        self._draw_aa_circle(
            img,
            dot_cx,
            dot_cy,
            dot_r,
            accent,
            "#ffffff",
        )

        return img

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        c = color.lstrip("#")
        if len(c) != 6:
            raise ValueError(f"Expected #RRGGBB color, got: {color}")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def _blend_hex(cls, fg: str, bg: str, alpha: float) -> str:
        a = max(0.0, min(1.0, alpha))
        fr, fg_c, fb = cls._hex_to_rgb(fg)
        br, bg_c, bb = cls._hex_to_rgb(bg)
        r = int((fr * a) + (br * (1.0 - a)))
        g = int((fg_c * a) + (bg_c * (1.0 - a)))
        b = int((fb * a) + (bb * (1.0 - a)))
        return cls._rgb_to_hex((r, g, b))

    def _find_fontawesome_font_path(self) -> Path | None:
        candidates: list[Path] = []

        try:
            import fontawesomefree
            base = Path(fontawesomefree.__file__).resolve().parent
            candidates.extend(
                [
                    base
                    / "static"
                    / "fontawesomefree"
                    / "js-packages"
                    / "@fortawesome"
                    / "fontawesome-free"
                    / "webfonts"
                    / "fa-solid-900.ttf",
                    base / "webfonts" / "fa-solid-900.ttf",
                ]
            )
        except Exception:
            base = None

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            mp = Path(meipass)
            candidates.extend(
                [
                    mp
                    / "fontawesomefree"
                    / "static"
                    / "fontawesomefree"
                    / "js-packages"
                    / "@fortawesome"
                    / "fontawesome-free"
                    / "webfonts"
                    / "fa-solid-900.ttf",
                    mp / "fontawesomefree" / "webfonts" / "fa-solid-900.ttf",
                    mp / "webfonts" / "fa-solid-900.ttf",
                ]
            )

        local_base = Path(__file__).resolve().parent
        candidates.extend(
            [
                local_base / "webfonts" / "fa-solid-900.ttf",
                local_base / "assets" / "webfonts" / "fa-solid-900.ttf",
            ]
        )
        try:
            import site

            site_paths = []
            site_paths.extend(site.getsitepackages())
            user_site = site.getusersitepackages()
            if user_site:
                site_paths.append(user_site)
            for sp in site_paths:
                root = Path(sp)
                candidates.extend(
                    [
                        root / "fontawesomefree" / "webfonts" / "fa-solid-900.ttf",
                        root
                        / "fontawesomefree"
                        / "static"
                        / "fontawesomefree"
                        / "js-packages"
                        / "@fortawesome"
                        / "fontawesome-free"
                        / "webfonts"
                        / "fa-solid-900.ttf",
                    ]
                )
        except Exception:
            pass

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except Exception:
                continue

        search_roots: list[Path] = []
        if base is not None:
            search_roots.append(base)
        if meipass:
            search_roots.append(Path(meipass))
        search_roots.append(local_base)

        for root in search_roots:
            try:
                found = next(root.rglob("fa-solid-900.ttf"))
                if found.exists():
                    return found
            except Exception:
                continue

        for entry in sys.path:
            try:
                root = Path(entry)
                if not root.exists() or not root.is_dir():
                    continue
                if "site-packages" not in str(root).lower():
                    continue
                found = next(root.rglob("fa-solid-900.ttf"))
                if found.exists():
                    return found
            except Exception:
                continue

        return None

    @staticmethod
    def _hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
        c = color.lstrip("#")
        if len(c) != 6:
            return (0, 0, 0, alpha)
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha)

    def _build_fa_action_icon(
        self,
        icon_name: str,
        bg_color: str,
        size: int,
        radius: int,
        circle: bool = False,
    ) -> tk.PhotoImage | None:
        icon_font_path = getattr(self, "icon_font_path", None)
        icon_cache = getattr(self, "action_icon_cache", None)
        if not isinstance(icon_cache, dict):
            icon_cache = {}
            self.action_icon_cache = icon_cache

        if (
            Image is None
            or ImageDraw is None
            or ImageFont is None
            or ImageTk is None
            or icon_font_path is None
        ):
            return None

        glyphs = {
            "mic": "\uf130",
            "stop": "\uf04d",
            "keyboard": "\uf11c",
        }
        glyph = glyphs.get(icon_name)
        if not glyph:
            return None

        cache_key = (icon_name, bg_color, size, radius)
        cached = icon_cache.get(cache_key)
        if cached is not None:
            return cached

        scale = 4
        px = size * scale
        pad = int(px * 0.14)
        image = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        shape_box = (pad, pad, px - pad, px - pad)
        fill = self._hex_to_rgba(bg_color)
        outline = self._hex_to_rgba(self._blend_hex(bg_color, "#000000", 0.12))
        if circle:
            draw.ellipse(shape_box, fill=fill, outline=outline, width=max(2, scale))
        else:
            draw.rounded_rectangle(
                shape_box,
                radius=max(4, radius * scale),
                fill=fill,
                outline=outline,
                width=max(2, scale),
            )

        font_px = int(size * 0.42 * scale)
        if icon_name == "keyboard":
            font_px = int(size * 0.40 * scale)
        font = ImageFont.truetype(str(icon_font_path), font_px)
        bbox = draw.textbbox((0, 0), glyph, font=font)
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        x = (px - glyph_w) / 2.0 - bbox[0]
        y = (px - glyph_h) / 2.0 - bbox[1]
        if icon_name == "mic":
            y -= int(scale * 0.5)
        draw.text((x, y), glyph, font=font, fill=(255, 255, 255, 255))

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((size, size), resample=resampling)
        tk_img = ImageTk.PhotoImage(image)
        icon_cache[cache_key] = tk_img
        return tk_img

    def _build_fa_nav_icon(self, icon_name: str, color: str, size: int = 18) -> tk.PhotoImage | None:
        icon_font_path = getattr(self, "icon_font_path", None)
        icon_cache = getattr(self, "action_icon_cache", None)
        if not isinstance(icon_cache, dict):
            icon_cache = {}
            self.action_icon_cache = icon_cache

        if (
            Image is None
            or ImageDraw is None
            or ImageFont is None
            or ImageTk is None
            or icon_font_path is None
        ):
            return None

        glyphs = {
            "home": "\uf015",
            "history": "\uf017",
            "dictionary": "\uf02d",
            "settings": "\uf013",
        }
        glyph = glyphs.get(icon_name)
        if not glyph:
            return None

        cache_key = ("nav", icon_name, color, size)
        cached = icon_cache.get(cache_key)
        if cached is not None:
            return cached

        scale = 4
        px = size * scale
        image = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font_px = int(size * 0.9 * scale)
        font = ImageFont.truetype(str(icon_font_path), font_px)
        bbox = draw.textbbox((0, 0), glyph, font=font)
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        x = (px - glyph_w) / 2.0 - bbox[0]
        y = (px - glyph_h) / 2.0 - bbox[1]
        draw.text((x, y), glyph, font=font, fill=self._hex_to_rgba(color))

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((size, size), resample=resampling)
        tk_img = ImageTk.PhotoImage(image)
        icon_cache[cache_key] = tk_img
        return tk_img

    def _build_fa_brand_icon(self, size: int, bg_color: str) -> tk.PhotoImage | None:
        icon_font_path = getattr(self, "icon_font_path", None)
        icon_cache = getattr(self, "action_icon_cache", None)
        if not isinstance(icon_cache, dict):
            icon_cache = {}
            self.action_icon_cache = icon_cache

        if (
            Image is None
            or ImageDraw is None
            or ImageFont is None
            or ImageTk is None
            or icon_font_path is None
        ):
            return None

        cache_key = ("brand", bg_color, str(size))
        cached = icon_cache.get(cache_key)
        if cached is not None:
            return cached

        scale = 6 if size <= 24 else 4
        px = size * scale
        image = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        outer_pad = int(px * 0.06)
        inner_pad = int(px * 0.10)
        draw.ellipse(
            (outer_pad, outer_pad, px - outer_pad, px - outer_pad),
            fill=self._hex_to_rgba("#1d4ed8"),
        )
        draw.ellipse(
            (inner_pad, inner_pad, px - inner_pad, px - inner_pad),
            fill=self._hex_to_rgba("#2563eb"),
        )

        glow_r = max(2, int(px * 0.16))
        glow_x = int(px * 0.36)
        glow_y = int(px * 0.34)
        draw.ellipse(
            (glow_x - glow_r, glow_y - glow_r, glow_x + glow_r, glow_y + glow_r),
            fill=self._hex_to_rgba("#60a5fa", 180),
        )

        mic_glyph = "\uf130"
        mic_font_px = int(size * (0.62 if size <= 24 else 0.52) * scale)
        font = ImageFont.truetype(str(icon_font_path), mic_font_px)
        bbox = draw.textbbox((0, 0), mic_glyph, font=font)
        gx = (px - (bbox[2] - bbox[0])) / 2.0 - bbox[0]
        gy = (px - (bbox[3] - bbox[1])) / 2.0 - bbox[1] - int(px * 0.01)
        draw.text((gx, gy), mic_glyph, font=font, fill=(255, 255, 255, 255))

        dot_r = max(3, int(px * 0.10))
        dot_cx = px - inner_pad - dot_r
        dot_cy = px - inner_pad - dot_r
        ring_r = dot_r + max(2, int(px * 0.02))
        draw.ellipse(
            (dot_cx - ring_r, dot_cy - ring_r, dot_cx + ring_r, dot_cy + ring_r),
            fill=(255, 255, 255, 255),
        )
        draw.ellipse(
            (dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r),
            fill=self._hex_to_rgba("#ef4444"),
        )

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((size, size), resample=resampling)
        tk_img = ImageTk.PhotoImage(image)
        icon_cache[cache_key] = tk_img
        return tk_img

    def _build_tray_icon_image(self, size: int = 64):
        if Image is None or ImageDraw is None:
            return None
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        outer_pad = int(size * 0.06)
        inner_pad = int(size * 0.11)
        draw.ellipse((outer_pad, outer_pad, size - outer_pad, size - outer_pad), fill=(29, 78, 216, 255))
        draw.ellipse((inner_pad, inner_pad, size - inner_pad, size - inner_pad), fill=(37, 99, 235, 255))
        glow_r = max(2, int(size * 0.16))
        glow_x = int(size * 0.36)
        glow_y = int(size * 0.34)
        draw.ellipse((glow_x - glow_r, glow_y - glow_r, glow_x + glow_r, glow_y + glow_r), fill=(96, 165, 250, 170))

        # Prefer FA mic glyph so tray icon stays consistent with app icon set.
        icon_font_path = getattr(self, "icon_font_path", None)
        if ImageFont is not None and icon_font_path is not None and Path(icon_font_path).exists():
            try:
                glyph = "\uf130"
                font_px = int(size * 0.5)
                font = ImageFont.truetype(str(icon_font_path), font_px)
                bbox = draw.textbbox((0, 0), glyph, font=font)
                gx = (size - (bbox[2] - bbox[0])) / 2.0 - bbox[0]
                gy = (size - (bbox[3] - bbox[1])) / 2.0 - bbox[1] - int(size * 0.02)
                draw.text((gx, gy), glyph, font=font, fill=(255, 255, 255, 255))
            except Exception:
                draw.ellipse((size * 0.42, size * 0.24, size * 0.58, size * 0.57), fill=(255, 255, 255, 255))
        else:
            draw.ellipse((size * 0.42, size * 0.24, size * 0.58, size * 0.57), fill=(255, 255, 255, 255))

        dot_r = max(3, int(size * 0.10))
        dot_cx = size - inner_pad - dot_r
        dot_cy = size - inner_pad - dot_r
        draw.ellipse((dot_cx - dot_r - 2, dot_cy - dot_r - 2, dot_cx + dot_r + 2, dot_cy + dot_r + 2), fill=(255, 255, 255, 255))
        draw.ellipse((dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r), fill=(239, 68, 68, 255))
        return image

    def _on_tray_open(self, _icon=None, _item=None) -> None:
        try:
            self.root.after(0, self._restore_from_tray)
        except Exception:
            pass

    def _on_tray_quit(self, _icon=None, _item=None) -> None:
        try:
            self.root.after(0, self._quit_from_tray)
        except Exception:
            pass

    def _ensure_tray_icon(self) -> bool:
        if pystray is None:
            return False
        if self.tray_icon is not None:
            return True

        tray_image = self._build_tray_icon_image(size=64)
        if tray_image is None:
            return False

        try:
            menu = pystray.Menu(
                pystray.MenuItem(f"Open {self.app.APP_NAME}", self._on_tray_open),
                pystray.MenuItem("Quit", self._on_tray_quit),
            )
            self.tray_icon = pystray.Icon(
                f"{self.app.APP_NAME}_tray",
                tray_image,
                self.app.APP_NAME,
                menu,
            )
            self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            self.tray_thread.start()
            return True
        except Exception:
            self.tray_icon = None
            self.tray_thread = None
            return False

    def _restore_from_tray(self) -> None:
        if self.closing:
            return
        try:
            self.root.state("normal")
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
            self.root.after(700, lambda: self.root.attributes("-topmost", False))
            if self.custom_titlebar_enabled:
                self.root.after(20, self._restore_custom_chrome)
            self.status_var.set("Ready")
        except Exception:
            pass

    def _hide_to_tray(self) -> None:
        if self.closing:
            return
        if not self._ensure_tray_icon():
            # Fallback: if tray is not available, preserve old close behavior.
            self.app.stop()
            self._close_ui()
            return
        try:
            self.root.withdraw()
            self.status_var.set(f"{self.app.APP_NAME} is running in tray")
        except Exception:
            pass

    def _shutdown_tray(self) -> None:
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.tray_icon = None
        self.tray_thread = None

    def _quit_from_tray(self) -> None:
        if self.closing:
            return
        self.app.stop()
        self._close_ui()

    def _draw_aa_circle(
        self,
        image: tk.PhotoImage,
        cx: float,
        cy: float,
        radius: float,
        color: str,
        bg_color: str,
    ) -> None:
        if radius <= 0:
            return
        edge = 1.0
        min_x = max(0, int(math.floor(cx - radius - edge)))
        max_x = min(image.width() - 1, int(math.ceil(cx + radius + edge)))
        min_y = max(0, int(math.floor(cy - radius - edge)))
        max_y = min(image.height() - 1, int(math.ceil(cy + radius + edge)))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                dx = (x + 0.5) - cx
                dy = (y + 0.5) - cy
                dist = math.sqrt((dx * dx) + (dy * dy))
                if dist <= (radius - edge):
                    px = color
                elif dist <= (radius + edge):
                    alpha = (radius + edge - dist) / (2.0 * edge)
                    px = self._blend_hex(color, bg_color, alpha)
                else:
                    continue
                image.put(px, to=(x, y, x + 1, y + 1))

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.c_surface,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        tone: str = "neutral",
        size: str = "md",
    ) -> tk.Button:
        if tone == "primary":
            bg, active, fg = self.c_primary, self.c_primary_hover, "#ffffff"
        elif tone == "success":
            bg, active, fg = self.c_success, self.c_success_hover, "#ffffff"
        elif tone == "danger":
            bg, active, fg = self.c_danger, self.c_danger_hover, "#ffffff"
        elif tone == "dark":
            bg, active, fg = self.c_dark, "#0f172a", "#ffffff"
        else:
            bg, active, fg = self.c_soft, "#2a3942", self.c_text

        pad_x = 14 if size == "md" else 10
        pad_y = 8 if size == "md" else 6
        font = ("Segoe UI", 10, "bold") if size == "md" else ("Segoe UI", 9, "bold")

        return tk.Button(
            parent,
            text=text,
            command=command,
            font=font,
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=pad_x,
            pady=pad_y,
            highlightthickness=0,
            cursor="hand2",
        )

    def _build_layout(self) -> None:
        root_wrap = tk.Frame(
            self.root,
            bg=self.c_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        root_wrap.pack(fill="both", expand=True)

        if self.custom_titlebar_enabled:
            self._build_custom_titlebar(root_wrap)

        body = tk.Frame(root_wrap, bg=self.c_bg)
        body.pack(fill="both", expand=True)

        shell = tk.Frame(body, bg=self.c_bg)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(
            shell,
            bg=self.c_sidebar,
            width=300,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        nav_wrap = tk.Frame(sidebar, bg=self.c_sidebar)
        nav_wrap.pack(fill="x", padx=10, pady=(16, 8))
        self._add_nav_button(nav_wrap, "home", "Home", "home")
        self._add_nav_button(nav_wrap, "history", "History", "history")
        self._add_nav_button(nav_wrap, "dictionary", "Dictionary", "dictionary")
        self._add_nav_button(nav_wrap, "settings", "Settings", "settings")

        tip = self._card(sidebar)
        tip.pack(side="bottom", fill="x", padx=10, pady=10)
        tk.Label(
            tip,
            text="Shortcuts",
            font=("Segoe UI", 13, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(
            tip,
            textvariable=self.hotkey_hint_var,
            font=("Segoe UI", 10),
            bg=self.c_surface,
            fg=self.c_muted,
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))
        ttk.Separator(tip, orient="horizontal").pack(fill="x", padx=0, pady=(0, 8))
        tk.Label(
            tip,
            textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        main = tk.Frame(shell, bg=self.c_bg)
        main.pack(side="left", fill="both", expand=True)

        content_root = tk.Frame(main, bg=self.c_bg)
        content_root.pack(side="left", fill="both", expand=True, padx=14, pady=(14, 12))

        self.page_frames["home"] = self._build_home_page(content_root)
        self.page_frames["history"] = self._build_history_page(content_root)
        self.page_frames["dictionary"] = self._build_dictionary_page(content_root)
        self.page_frames["settings"] = self._build_settings_page(content_root)

        for frame in self.page_frames.values():
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._show_page("home")

    def _build_custom_titlebar(self, parent: tk.Widget) -> None:
        bar = tk.Frame(
            parent,
            bg="#1f1f1f",
            height=40,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        bar.pack(fill="x")
        bar.pack_propagate(False)

        left = tk.Frame(bar, bg="#1f1f1f")
        left.pack(side="left", fill="y")

        tk.Label(left, image=self.brand_icon, bg="#1f1f1f").pack(side="left", padx=(12, 8))
        tk.Label(
            left,
            text=self.app.APP_NAME,
            font=("Segoe UI", 10, "bold"),
            bg="#1f1f1f",
            fg=self.c_text,
        ).pack(side="left", pady=(1, 0))

        controls = tk.Frame(bar, bg="#1f1f1f")
        controls.pack(side="right", fill="y")

        btn_min = self._create_titlebar_control_button(
            controls,
            text="\ue921",
            command=self._minimize_from_titlebar,
            close_button=False,
        )
        btn_min.pack(side="left", fill="y")

        self.titlebar_max_btn = self._create_titlebar_control_button(
            controls,
            text="\ue922",
            command=self._toggle_maximize_restore,
            close_button=False,
        )
        self.titlebar_max_btn.pack(side="left", fill="y")
        self._sync_titlebar_max_icon()

        btn_close = self._create_titlebar_control_button(
            controls,
            text="\ue8bb",
            command=self._on_close,
            close_button=True,
        )
        btn_close.pack(side="left", fill="y")

        drag_widgets = [bar, left]
        for widget in drag_widgets:
            widget.bind("<ButtonPress-1>", self._titlebar_start_drag)
            widget.bind("<B1-Motion>", self._titlebar_drag)
            widget.bind("<Double-Button-1>", lambda _e: self._toggle_maximize_restore())

    def _create_titlebar_control_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        close_button: bool,
    ) -> tk.Button:
        bg = "#1f1f1f"
        fg = "#f2f2f2"
        hover_bg = "#2c2c2c"
        close_hover_bg = "#e81123"
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe MDL2 Assets", 10),
            bg=bg,
            fg=fg,
            activebackground=close_hover_bg if close_button else hover_bg,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            width=4,
            padx=0,
            pady=0,
            cursor="hand2",
            highlightthickness=0,
            takefocus=False,
        )
        btn.bind(
            "<Enter>",
            lambda _e, b=btn, c=close_button: b.configure(
                bg=close_hover_bg if c else hover_bg,
                fg="#ffffff",
            ),
        )
        btn.bind("<Leave>", lambda _e, b=btn: b.configure(bg=bg, fg=fg))
        return btn

    def _sync_titlebar_max_icon(self) -> None:
        if self.titlebar_max_btn is None:
            return
        self.titlebar_max_btn.configure(text="\ue923" if self.window_is_maximized else "\ue922")

    def _titlebar_start_drag(self, event) -> None:
        if self.window_is_maximized:
            return
        self._drag_start_x = int(event.x_root - self.root.winfo_x())
        self._drag_start_y = int(event.y_root - self.root.winfo_y())

    def _titlebar_drag(self, event) -> None:
        if self.window_is_maximized:
            return
        x = int(event.x_root - self._drag_start_x)
        y = int(event.y_root - self._drag_start_y)
        try:
            self.root.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _minimize_from_titlebar(self) -> None:
        try:
            if self.custom_titlebar_enabled:
                self.root.overrideredirect(False)
            self.root.iconify()
        except Exception:
            pass

    def _toggle_maximize_restore(self) -> None:
        if self.window_is_maximized:
            try:
                self.root.state("normal")
            except Exception:
                pass
            if self.window_restore_geometry:
                try:
                    self.root.geometry(self.window_restore_geometry)
                except Exception:
                    pass
            self.window_is_maximized = False
        else:
            try:
                self.window_restore_geometry = self.root.geometry()
            except Exception:
                self.window_restore_geometry = ""
            try:
                self.root.state("zoomed")
            except Exception:
                sw = self.root.winfo_screenwidth()
                sh = self.root.winfo_screenheight()
                self.root.geometry(f"{sw}x{sh}+0+0")
            self.window_is_maximized = True

        self._sync_titlebar_max_icon()

    def _on_root_map(self, _event=None) -> None:
        if self.closing or not self.custom_titlebar_enabled:
            return
        try:
            self.window_is_maximized = str(self.root.state()) == "zoomed"
        except Exception:
            pass
        self._sync_titlebar_max_icon()
        self.root.after(10, self._restore_custom_chrome)

    def _restore_custom_chrome(self) -> None:
        if self.closing or not self.custom_titlebar_enabled:
            return
        try:
            self.root.overrideredirect(True)
        except Exception:
            pass

    def _on_alt_f4(self, _event=None) -> str:
        self._on_close()
        return "break"

    def _add_rail_button(self, parent: tk.Widget, page: str, icon_name: str) -> None:
        inactive_icon = self._build_fa_nav_icon(icon_name, self.c_muted, size=17)
        active_icon = self._build_fa_nav_icon(icon_name, self.c_text, size=17)
        self.rail_button_icons[page] = (inactive_icon, active_icon)

        btn = tk.Button(
            parent,
            text="",
            relief="flat",
            bd=0,
            width=3,
            height=1,
            bg="#111b21",
            activebackground="#2a3942",
            cursor="hand2",
            highlightthickness=0,
            command=lambda p=page: self._show_page(p),
        )
        if inactive_icon is not None:
            btn.configure(image=inactive_icon)
        btn.pack(pady=4)
        self.rail_buttons[page] = btn

    def _add_nav_button(self, parent: tk.Widget, page: str, label: str, icon_name: str) -> None:
        inactive_icon = self._build_fa_nav_icon(icon_name, self.c_muted, size=18)
        active_icon = self._build_fa_nav_icon(icon_name, self.c_text, size=18)
        self.nav_button_icons[page] = (inactive_icon, active_icon)

        fallback_glyphs = {
            "home": "\u2302",
            "history": "\u25CB",
            "dictionary": "\u2261",
            "settings": "\u2699",
        }
        text_prefix = "" if inactive_icon is not None else f"{fallback_glyphs.get(icon_name, '')}  "
        btn = tk.Button(
            parent,
            text=f"{text_prefix}{label}",
            relief="flat",
            font=("Segoe UI", 13),
            anchor="w",
            padx=14,
            pady=10,
            bg=self.c_sidebar,
            activebackground=self.c_soft,
            fg=self.c_muted,
            activeforeground=self.c_text,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            command=lambda p=page: self._show_page(p),
        )
        if inactive_icon is not None:
            btn.configure(image=inactive_icon, compound="left")
        btn.pack(fill="x", padx=0, pady=4)
        self.nav_buttons[page] = btn

    def _show_page(self, page: str) -> None:
        frame = self.page_frames.get(page)
        if frame is None:
            return
        frame.tkraise()
        page_titles = {
            "home": "Home",
            "history": "History",
            "dictionary": "Dictionary",
            "settings": "Settings",
        }
        self.page_title_var.set(page_titles.get(page, "lesstyper"))
        for name, button in self.nav_buttons.items():
            inactive_icon, active_icon = self.nav_button_icons.get(name, (None, None))
            if name == page:
                button.configure(bg=self.c_soft, fg=self.c_text, font=("Segoe UI", 14, "bold"))
                if active_icon is not None:
                    button.configure(image=active_icon)
            else:
                button.configure(bg=self.c_sidebar, fg=self.c_muted, font=("Segoe UI", 14))
                if inactive_icon is not None:
                    button.configure(image=inactive_icon)

    def _build_home_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.c_bg)

        hero = self._card(page)
        hero.pack(fill="both", expand=True)

        header = tk.Frame(hero, bg=self.c_surface)
        header.pack(fill="x", padx=18, pady=(12, 10))
        tk.Label(
            header,
            text="Voice Workspace",
            font=("Segoe UI", 16, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(side="left")
        tk.Label(
            header,
            text="Speak naturally, turn voice into text in any app",
            font=("Segoe UI", 10),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(side="left", padx=(14, 0), pady=(2, 0))

        actions = tk.Frame(hero, bg=self.c_surface)
        actions.pack(fill="x", padx=18, pady=(4, 10))

        start_wrap = tk.Frame(actions, bg=self.c_surface)
        start_wrap.pack(side="left", padx=(2, 0))
        self.home_start_canvas = tk.Canvas(
            start_wrap,
            width=78,
            height=78,
            bg=self.c_surface,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.home_start_canvas.pack(anchor="center")
        self.home_start_canvas.bind("<Button-1>", lambda _e: self.app.toggle_recording())
        self._render_home_start_icon(listening=False)
        self.home_toggle_caption = tk.Label(
            start_wrap,
            text="Start",
            font=("Segoe UI", 12),
            bg=self.c_surface,
            fg=self.c_muted,
        )
        self.home_toggle_caption.pack(anchor="center", pady=(5, 0))

        tk.Frame(actions, bg=self.c_border, width=1, height=54).pack(
            side="left", padx=18, pady=(10, 0)
        )

        stop_wrap = tk.Frame(actions, bg=self.c_surface)
        stop_wrap.pack(side="left")
        stop_canvas = tk.Canvas(
            stop_wrap,
            width=78,
            height=78,
            bg=self.c_surface,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        stop_canvas.pack(anchor="center")
        stop_canvas.bind("<Button-1>", lambda _e: self.app.stop_recording())
        self._render_home_stop_icon(stop_canvas)
        tk.Label(
            stop_wrap,
            text="Stop",
            font=("Segoe UI", 12),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(anchor="center", pady=(5, 0))

        tk.Frame(actions, bg=self.c_border, width=1, height=54).pack(
            side="left", padx=18, pady=(10, 0)
        )

        shortcut_wrap = tk.Frame(actions, bg=self.c_surface)
        shortcut_wrap.pack(side="left")
        shortcut_canvas = tk.Canvas(
            shortcut_wrap,
            width=78,
            height=78,
            bg=self.c_surface,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        shortcut_canvas.pack(anchor="center")
        shortcut_canvas.bind("<Button-1>", lambda _e: self._show_page("settings"))
        self._render_home_shortcut_icon(shortcut_canvas)
        tk.Label(
            shortcut_wrap,
            text="Shortcuts",
            font=("Segoe UI", 12),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(anchor="center", pady=(5, 0))

        stats = tk.Frame(page, bg=self.c_bg)
        stats.pack(fill="x", pady=(12, 12))
        self._stat_card(stats, "Words dictated", self.words_var).pack(
            side="left", fill="x", expand=True, padx=(0, 6)
        )
        self._stat_card(stats, "Total time saved", self.time_saved_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        self._stat_card(stats, "Average WPM", self.wpm_var).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )

        latest = self._card(page)
        latest.pack(fill="both", expand=True)
        top = tk.Frame(latest, bg=self.c_surface)
        top.pack(fill="x", padx=18, pady=(14, 10))
        tk.Label(
            top,
            text="Latest Transcription",
            font=("Segoe UI", 15, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(side="left")

        bubble_wrap = tk.Frame(latest, bg=self.c_surface)
        bubble_wrap.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        bubble = tk.Frame(
            bubble_wrap,
            bg="#1f2c33",
            bd=0,
            highlightthickness=1,
            highlightbackground="#2f464f",
        )
        bubble.pack(fill="both", expand=True, anchor="w")
        self.latest_text_label = tk.Label(
            bubble,
            textvariable=self.last_text_var,
            font=("Segoe UI", 20),
            bg="#1f2c33",
            fg=self.c_text,
            wraplength=980,
            justify="left",
            anchor="nw",
        )
        self.latest_text_label.pack(fill="both", expand=True, padx=16, pady=14)

        return page

    def _on_home_hero_configure(self, event) -> None:
        if self.home_title_label is None:
            return
        width = max(320, int(getattr(event, "width", 0) or 0))
        if width <= 0:
            return

        # Responsive heading: shrink on smaller windows and wrap before clipping.
        size = max(22, min(44, int(width / 23)))
        self.home_title_label.configure(
            font=("Segoe UI", size, "bold"),
            wraplength=max(320, width - 72),
        )

    def _draw_rounded_rect(
        self,
        canvas: tk.Canvas,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        fill: str,
        outline: str = "",
        width: int = 1,
    ) -> None:
        r = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        canvas.create_rectangle(
            x1 + r,
            y1,
            x2 - r,
            y2,
            fill=fill,
            outline=outline,
            width=width,
        )
        canvas.create_rectangle(
            x1,
            y1 + r,
            x2,
            y2 - r,
            fill=fill,
            outline=outline,
            width=width,
        )
        canvas.create_oval(
            x1,
            y1,
            x1 + (2 * r),
            y1 + (2 * r),
            fill=fill,
            outline=outline,
            width=width,
        )
        canvas.create_oval(
            x2 - (2 * r),
            y1,
            x2,
            y1 + (2 * r),
            fill=fill,
            outline=outline,
            width=width,
        )
        canvas.create_oval(
            x1,
            y2 - (2 * r),
            x1 + (2 * r),
            y2,
            fill=fill,
            outline=outline,
            width=width,
        )
        canvas.create_oval(
            x2 - (2 * r),
            y2 - (2 * r),
            x2,
            y2,
            fill=fill,
            outline=outline,
            width=width,
        )

    def _render_home_start_icon(self, listening: bool) -> None:
        if self.home_start_canvas is None:
            return

        c = self.home_start_canvas
        c.delete("all")

        w = int(c.cget("width"))
        h = int(c.cget("height"))
        cx = w // 2
        cy = h // 2
        pad = 13

        icon_img = self._build_fa_action_icon(
            icon_name="stop" if listening else "mic",
            bg_color=self.c_danger if listening else self.c_success,
            size=min(w, h),
            radius=16,
            circle=not listening,
        )
        if icon_img is not None:
            c.create_image(cx, cy, image=icon_img)
            c._icon_image = icon_img
            return

        if listening:
            self._draw_rounded_rect(
                c,
                pad,
                pad,
                w - pad,
                h - pad,
                16,
                self.c_danger,
                outline="#cc4a4d",
                width=1,
            )
            self._draw_rounded_rect(
                c,
                cx - 9,
                cy - 9,
                cx + 9,
                cy + 9,
                4,
                "#ffffff",
            )
            return

        # Start icon: teal circular badge + white mic glyph.
        c.create_oval(
            pad,
            pad,
            w - pad,
            h - pad,
            fill=self.c_success,
            outline="#2b9a99",
            width=1,
        )
        c.create_oval(cx - 6, cy - 17, cx + 6, cy + 1, fill="#ffffff", outline="")
        c.create_rectangle(cx - 6, cy - 10, cx + 6, cy + 7, fill="#ffffff", outline="")
        c.create_arc(
            cx - 16,
            cy + 1,
            cx + 16,
            cy + 23,
            start=200,
            extent=140,
            style="arc",
            outline="#ffffff",
            width=2,
        )
        c.create_line(cx, cy + 6, cx, cy + 16, fill="#ffffff", width=2)

    def _render_home_stop_icon(self, canvas: tk.Canvas) -> None:
        canvas.delete("all")
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
        cx = w // 2
        cy = h // 2
        pad = 13

        icon_img = self._build_fa_action_icon(
            icon_name="stop",
            bg_color=self.c_danger,
            size=min(w, h),
            radius=16,
        )
        if icon_img is not None:
            canvas.create_image(cx, cy, image=icon_img)
            canvas._icon_image = icon_img
            return

        self._draw_rounded_rect(
            canvas,
            pad,
            pad,
            w - pad,
            h - pad,
            16,
            self.c_danger,
            outline="#cc4a4d",
            width=1,
        )
        self._draw_rounded_rect(canvas, cx - 9, cy - 9, cx + 9, cy + 9, 4, "#ffffff")

    def _render_home_shortcut_icon(self, canvas: tk.Canvas) -> None:
        canvas.delete("all")
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
        cx = w // 2
        cy = h // 2
        pad = 13

        icon_img = self._build_fa_action_icon(
            icon_name="keyboard",
            bg_color="#2a3942",
            size=min(w, h),
            radius=12,
        )
        if icon_img is not None:
            canvas.create_image(cx, cy, image=icon_img)
            canvas._icon_image = icon_img
            return

        self._draw_rounded_rect(
            canvas,
            pad,
            pad,
            w - pad,
            h - pad,
            12,
            "#2a3942",
            outline="#31434d",
            width=1,
        )
        self._draw_rounded_rect(canvas, cx - 15, cy - 9, cx + 15, cy + 11, 4, "#ffffff")
        key_w = 4
        key_h = 3
        gap = 2
        start_x = cx - 11
        start_y = cy - 5
        for row in range(3):
            for col in range(5):
                x1 = start_x + col * (key_w + gap)
                y1 = start_y + row * (key_h + gap)
                x2 = x1 + key_w
                y2 = y1 + key_h
                self._draw_rounded_rect(
                    canvas,
                    x1,
                    y1,
                    x2,
                    y2,
                    1,
                    "#2a3942",
                )

    def _build_history_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.c_bg)

        wrap = self._card(page)
        wrap.pack(fill="both", expand=True)

        header = tk.Frame(wrap, bg=self.c_surface)
        header.pack(fill="x", padx=20, pady=(16, 8))
        tk.Label(
            header,
            text="History",
            font=("Segoe UI", 18, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(side="left")
        self._button(header, "Copy All", self._copy_history, tone="primary", size="sm").pack(
            side="right"
        )
        self._button(
            header,
            "Clear History",
            self.app.clear_history,
            tone="danger",
            size="sm",
        ).pack(side="right", padx=8)

        self.history_text = tk.Text(
            wrap,
            bg=self.c_soft,
            fg=self.c_text,
            relief="flat",
            font=self.default_mono_font,
            bd=0,
        )
        self.history_text.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        self.history_text.configure(state="disabled")

        return page

    def _build_dictionary_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.c_bg)

        wrap = self._card(page)
        wrap.pack(fill="both", expand=True)

        tk.Label(
            wrap,
            text="Dictionary",
            font=("Segoe UI", 18, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w", padx=20, pady=(16, 4))
        tk.Label(
            wrap,
            text="Replace spoken words with your preferred terms, product names, or shortcuts.",
            font=("Segoe UI", 10),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(anchor="w", padx=20, pady=(0, 10))

        form = tk.Frame(wrap, bg=self.c_surface)
        form.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(form, text="Spoken", font=("Segoe UI", 10, "bold"), bg=self.c_surface, fg=self.c_text).grid(row=0, column=0, sticky="w")
        source_entry = tk.Entry(
            form,
            textvariable=self.dictionary_source_var,
            font=("Segoe UI", 10),
            bd=0,
            relief="flat",
            bg=self.c_surface,
            fg=self.c_text,
            highlightthickness=1,
            highlightbackground=self.c_border,
            highlightcolor=self.c_primary,
            insertbackground=self.c_text,
        )
        source_entry.grid(row=1, column=0, sticky="we", padx=(0, 12))
        source_entry.bind("<Return>", self._on_dictionary_enter)
        self.dictionary_source_entry = source_entry

        tk.Label(form, text="Replace with", font=("Segoe UI", 10, "bold"), bg=self.c_surface, fg=self.c_text).grid(row=0, column=1, sticky="w")
        target_entry = tk.Entry(
            form,
            textvariable=self.dictionary_target_var,
            font=("Segoe UI", 10),
            bd=0,
            relief="flat",
            bg=self.c_surface,
            fg=self.c_text,
            highlightthickness=1,
            highlightbackground=self.c_border,
            highlightcolor=self.c_primary,
            insertbackground=self.c_text,
        )
        target_entry.grid(row=1, column=1, sticky="we", padx=(0, 12))
        target_entry.bind("<Return>", self._on_dictionary_enter)
        self.dictionary_target_entry = target_entry

        self._button(form, "Add Rule", self._add_dictionary_rule, tone="primary", size="sm").grid(
            row=1, column=2, sticky="w"
        )

        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        table_wrap = tk.Frame(wrap, bg=self.c_surface)
        table_wrap.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.dictionary_tree = ttk.Treeview(
            table_wrap,
            columns=("spoken", "replacement"),
            show="headings",
            height=10,
            style="LessTyper.Treeview",
        )
        self.dictionary_tree.heading("spoken", text="Spoken")
        self.dictionary_tree.heading("replacement", text="Replacement")
        self.dictionary_tree.column("spoken", width=280)
        self.dictionary_tree.column("replacement", width=520)
        self.dictionary_tree.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.dictionary_tree.yview)
        scroll.pack(side="right", fill="y")
        self.dictionary_tree.configure(yscrollcommand=scroll.set)

        buttons = tk.Frame(wrap, bg=self.c_surface)
        buttons.pack(fill="x", padx=20, pady=(0, 18))
        self._button(
            buttons,
            "Remove Selected",
            self._remove_selected_dictionary_rule,
            tone="danger",
            size="sm",
        ).pack(side="left")
        self._button(
            buttons,
            "Clear All",
            self._clear_dictionary_rules,
            tone="neutral",
            size="sm",
        ).pack(side="left", padx=8)
        self._button(
            buttons,
            "Save Dictionary",
            self._save_dictionary_rules,
            tone="success",
            size="sm",
        ).pack(side="left")
        self._button(
            buttons,
            "Reload",
            self._reload_dictionary_tree,
            tone="dark",
            size="sm",
        ).pack(side="left", padx=8)

        return page
    def _build_settings_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.c_bg)

        wrap = self._card(page)
        wrap.pack(fill="both", expand=True)

        header = tk.Frame(wrap, bg=self.c_surface)
        header.pack(fill="x", padx=20, pady=(16, 8))
        tk.Label(
            header,
            text="Settings",
            font=("Segoe UI", 21, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Configure hotkeys, transcription model, and app behavior.",
            font=("Segoe UI", 10),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(wrap, bg=self.c_surface)
        body.pack(fill="both", expand=True, padx=20, pady=(6, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        transcription = self._card(body)
        transcription.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            transcription,
            text="Transcription",
            font=("Segoe UI", 13, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w", padx=14, pady=(12, 8))
        form = tk.Frame(transcription, bg=self.c_surface)
        form.pack(fill="x", padx=14, pady=(0, 12))
        form.grid_columnconfigure(1, weight=1)
        form.grid_columnconfigure(3, weight=1)

        self._field(form, 0, "Start/Stop Hotkey", self.start_hotkey_var, col=0)
        self._field(form, 0, "Quit Hotkey", self.quit_hotkey_var, col=2)
        self._field(form, 1, "Min Seconds", self.min_seconds_var, col=2)

        tk.Label(
            form,
            text="Language",
            font=("Segoe UI", 10, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).grid(row=1, column=0, sticky="w", padx=(20, 6), pady=(4, 6))

        language_combo = ttk.Combobox(
            form,
            textvariable=self.language_var,
            values=[LANGUAGE_DROPDOWN_LABELS["en"]],
            state="readonly",
            style="LessTyper.TCombobox",
            width=24,
        )
        language_combo.grid(row=1, column=1, sticky="we", padx=(0, 14), pady=(4, 6))
        language_combo.bind("<<ComboboxSelected>>", self._on_language_selection_changed)

        tk.Label(
            form,
            text="Model",
            font=("Segoe UI", 10, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).grid(row=2, column=0, sticky="w", padx=(20, 6), pady=(6, 8))
        ttk.Combobox(
            form,
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            state="readonly",
            style="LessTyper.TCombobox",
            width=24,
        ).grid(row=2, column=1, sticky="we", padx=(0, 14), pady=(6, 8))

        behavior = self._card(body)
        behavior.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(
            behavior,
            text="Behavior",
            font=("Segoe UI", 13, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w", padx=14, pady=(12, 8))

        options = tk.Frame(behavior, bg=self.c_surface)
        options.pack(fill="x", padx=14, pady=(0, 8))
        options.grid_columnconfigure(0, weight=1)
        self._styled_checkbox_row(
            options,
            row=0,
            text="Add trailing space after insert",
            variable=self.trailing_space_var,
            column=0,
            columnspan=1,
            padx=(0, 0),
            pady=(0, 6),
        )
        self._styled_checkbox_row(
            options,
            row=1,
            text="Remove filler words (um, uh)",
            variable=self.remove_fillers_var,
            column=0,
            columnspan=1,
            padx=(0, 0),
            pady=(0, 6),
        )
        self._styled_checkbox_row(
            options,
            row=2,
            text="Start with Windows",
            variable=self.start_with_windows_var,
            column=0,
            columnspan=1,
            padx=(0, 0),
            pady=(0, 10),
        )

        self.bengali_font_status_label = tk.Label(
            options,
            textvariable=self.bengali_font_status_var,
            font=("Segoe UI", 9, "bold"),
            bg=self.c_surface,
            fg="#065f46",
            wraplength=450,
            justify="left",
        )
        self.bengali_font_status_label.grid(
            row=3,
            column=0,
            sticky="w",
            padx=(0, 0),
            pady=(2, 6),
        )

        actions = tk.Frame(behavior, bg=self.c_surface)
        actions.pack(fill="x", padx=14, pady=(2, 14))
        self._button(actions, "Apply Settings", self._apply_settings, tone="primary").pack(
            side="left"
        )

        return page

    def _find_available_bengali_fonts(self) -> list[str]:
        try:
            available = set(tkfont.families(self.root))
        except Exception:
            return []

        return [family for family in BENGALI_FONT_CANDIDATES if family in available]

    def _is_bengali_language(self) -> bool:
        return normalize_language_input(self.app.config.language) == "bn"

    def _on_language_selection_changed(self, _event=None) -> None:
        selected_code = normalize_language_input(self.language_var.get())
        if selected_code not in ENABLED_LANGUAGE_CODES:
            self.language_var.set(language_dropdown_value(self.app.config.language))
            self.status_var.set(
                f"{language_display_name(selected_code)} support is coming soon. English is available now."
            )
            selected_code = normalize_language_input(self.language_var.get())

        self._update_bengali_font_status(selected_language=selected_code, push_status=False)

    def _update_bengali_font_status(
        self,
        selected_language: str | None = None,
        push_status: bool = False,
    ) -> None:
        language = normalize_language_input(selected_language or self.app.config.language)

        if language != "bn":
            self.bengali_font_status_var.set("")
            if self.bengali_font_status_label is not None:
                self.bengali_font_status_label.grid_remove()
            return

        if self.bengali_font_status_label is not None:
            self.bengali_font_status_label.grid()

        if self.available_bengali_fonts:
            primary = self.available_bengali_fonts[0]
            extra_count = len(self.available_bengali_fonts) - 1
            if extra_count > 0:
                message = (
                    f"Bengali font ready: {primary} (+{extra_count} more detected)."
                )
            else:
                message = f"Bengali font ready: {primary}."
            color = "#065f46"
        else:
            message = (
                "Warning: No Bengali font detected. Install Nirmala UI, SolaimanLipi, "
                "or Kalpurush for reliable Bangla rendering."
            )
            color = "#b91c1c"
            if push_status and language == "bn":
                self.status_var.set(message)

        self.bengali_font_status_var.set(message)
        if self.bengali_font_status_label is not None:
            self.bengali_font_status_label.configure(fg=color)

    def _apply_language_fonts(self) -> None:
        if self._is_bengali_language():
            latest_font = self.bengali_text_font
            history_font = self.bengali_mono_font
            tree_font = self.bengali_tree_font
            heading_font = self.bengali_text_font
        else:
            latest_font = self.default_text_font
            history_font = self.default_mono_font
            tree_font = self.default_tree_font
            heading_font = self.default_tree_heading_font

        if self.latest_text_label is not None:
            self.latest_text_label.configure(font=latest_font)
        if self.history_text is not None:
            self.history_text.configure(font=history_font)

        self.tree_style.configure(
            "LessTyper.Treeview",
            font=tree_font,
            rowheight=28,
            background=self.c_surface,
            fieldbackground=self.c_surface,
            foreground=self.c_text,
            bordercolor=self.c_border,
        )
        self.tree_style.configure(
            "LessTyper.Treeview.Heading",
            font=heading_font,
            background=self.c_soft,
            foreground=self.c_text,
            bordercolor=self.c_border,
        )

    def _stat_card(self, parent: tk.Widget, title: str, value_var: tk.StringVar) -> tk.Frame:
        card = self._card(parent)
        tk.Label(
            card,
            textvariable=value_var,
            font=("Segoe UI", 40, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 14),
            bg=self.c_surface,
            fg=self.c_muted,
        ).pack(anchor="w", padx=18, pady=(0, 16))
        return card

    def _styled_checkbox_row(
        self,
        parent: tk.Widget,
        row: int,
        text: str,
        variable: tk.BooleanVar,
        column: int = 2,
        columnspan: int = 2,
        padx: tuple[int, int] = (10, 20),
        pady: tuple[int, int] = (0, 8),
    ) -> None:
        row_wrap = tk.Frame(parent, bg=self.c_surface, cursor="hand2")
        row_wrap.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="we",
            padx=padx,
            pady=pady,
        )
        row_wrap.grid_columnconfigure(1, weight=1)

        box_canvas = tk.Canvas(
            row_wrap,
            width=22,
            height=22,
            bg=self.c_surface,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        box_canvas.grid(row=0, column=0, sticky="w")

        label = tk.Label(
            row_wrap,
            text=text,
            font=("Segoe UI", 10),
            bg=self.c_surface,
            fg=self.c_text,
            anchor="w",
            cursor="hand2",
        )
        label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        normal_bg = self.c_surface
        hover_bg = "#132028"

        def paint_checkbox() -> None:
            checked = bool(variable.get())
            box_canvas.delete("all")
            if checked:
                box_canvas.create_rectangle(
                    2,
                    2,
                    20,
                    20,
                    fill=self.c_primary,
                    outline=self.c_primary,
                    width=1,
                )
            else:
                box_canvas.create_rectangle(
                    2,
                    2,
                    20,
                    20,
                    fill="#10212b",
                    outline="#4e6471",
                    width=1,
                )
                box_canvas.create_rectangle(
                    4,
                    4,
                    18,
                    18,
                    fill="",
                    outline="#1f3541",
                    width=1,
                )
            if checked:
                box_canvas.create_line(
                    6,
                    11,
                    10,
                    15,
                    fill="#ffffff",
                    width=2,
                    capstyle="round",
                    joinstyle="round",
                )
                box_canvas.create_line(
                    10,
                    15,
                    16,
                    7,
                    fill="#ffffff",
                    width=2,
                    capstyle="round",
                    joinstyle="round",
                )

        def toggle(_event=None) -> str:
            variable.set(not bool(variable.get()))
            return "break"

        def set_hover(enabled: bool) -> None:
            bg = hover_bg if enabled else normal_bg
            row_wrap.configure(bg=bg)
            box_canvas.configure(bg=bg)
            label.configure(bg=bg)

        for widget in (row_wrap, box_canvas, label):
            widget.bind("<Button-1>", toggle)
            widget.bind("<Enter>", lambda _e: set_hover(True))
            widget.bind("<Leave>", lambda _e: set_hover(False))

        variable.trace_add("write", lambda *_args: paint_checkbox())
        paint_checkbox()

    def _field(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        var: tk.StringVar,
        col: int = 0,
    ) -> None:
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10, "bold"),
            bg=self.c_surface,
            fg=self.c_text,
        ).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=(4, 6))
        entry = tk.Entry(
            parent,
            textvariable=var,
            font=("Segoe UI", 10),
            bd=0,
            relief="flat",
            bg=self.c_surface,
            fg=self.c_text,
            highlightthickness=1,
            highlightbackground=self.c_border,
            highlightcolor=self.c_primary,
            insertbackground=self.c_text,
        )
        entry.grid(
            row=row,
            column=col + 1,
            sticky="we",
            padx=(0, 10),
            pady=(4, 6),
        )

    def _apply_settings(self) -> None:
        try:
            min_seconds = float(self.min_seconds_var.get().strip())
        except ValueError:
            self.status_var.set("Min seconds must be a number")
            return

        normalized_language = normalize_language_input(self.language_var.get())
        compatible_model, _ = ensure_model_language_compatibility(
            self.model_var.get(),
            normalized_language,
        )

        ok, msg = self.app.apply_settings(
            start_hotkey=self.start_hotkey_var.get(),
            quit_hotkey=self.quit_hotkey_var.get(),
            language=self.language_var.get(),
            min_seconds=min_seconds,
            trailing_space=self.trailing_space_var.get(),
            remove_fillers=self.remove_fillers_var.get(),
            model_size=self.model_var.get(),
        )

        if ok:
            startup_before = self.app.is_start_with_windows_enabled()
            startup_wanted = bool(self.start_with_windows_var.get())
            if startup_wanted != startup_before:
                startup_ok, startup_msg = self.app.set_start_with_windows(startup_wanted)
                if startup_ok:
                    msg = f"{msg} {startup_msg}"
                else:
                    self.start_with_windows_var.set(startup_before)
                    msg = f"{msg} {startup_msg}"

        self.status_var.set(msg)
        if ok:
            self.language_var.set(language_dropdown_value(self.app.config.language))
            self.model_var.set(compatible_model)
            self._apply_language_fonts()
            self._update_bengali_font_status(push_status=True)
            self._refresh_hotkeys_hint()
        else:
            self.language_var.set(language_dropdown_value(self.app.config.language))
            self._update_bengali_font_status(push_status=False)

    def _add_dictionary_rule(self) -> None:
        source = self.dictionary_source_var.get().strip()
        replacement = self.dictionary_target_var.get().strip()
        if not source or not replacement:
            self.status_var.set("Dictionary entries need both Spoken and Replace with")
            return
        if self.dictionary_tree is None:
            return

        existing = None
        for row_id in self.dictionary_tree.get_children():
            values = self.dictionary_tree.item(row_id, "values")
            if str(values[0]).strip().lower() == source.lower():
                existing = row_id
                break

        if existing is not None:
            self.dictionary_tree.item(existing, values=(source, replacement))
        else:
            self.dictionary_tree.insert("", "end", values=(source, replacement))

        self.dictionary_source_var.set("")
        self.dictionary_target_var.set("")
        self._save_dictionary_rules()
        if self.dictionary_source_entry is not None:
            self.dictionary_source_entry.focus_set()

    def _remove_selected_dictionary_rule(self) -> None:
        if self.dictionary_tree is None:
            return
        selected = self.dictionary_tree.selection()
        if not selected:
            self.status_var.set("Select a dictionary row to remove")
            return
        for row_id in selected:
            self.dictionary_tree.delete(row_id)
        self._save_dictionary_rules()

    def _clear_dictionary_rules(self) -> None:
        if self.dictionary_tree is None:
            return
        for row_id in self.dictionary_tree.get_children():
            self.dictionary_tree.delete(row_id)
        self._save_dictionary_rules()

    def _collect_dictionary_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        if self.dictionary_tree is None:
            return rows
        for row_id in self.dictionary_tree.get_children():
            values = self.dictionary_tree.item(row_id, "values")
            if len(values) != 2:
                continue
            src = str(values[0]).strip()
            dst = str(values[1]).strip()
            if src and dst:
                rows.append((src, dst))
        return rows

    def _on_dictionary_enter(self, _event=None) -> str:
        self._add_dictionary_rule()
        return "break"

    def _estimate_time_saved_minutes(self, words: int, dictation_seconds: float) -> float:
        # Conservative baseline typing speed for desktop text entry.
        baseline_typing_wpm = 40.0
        typing_seconds = (max(0, words) / baseline_typing_wpm) * 60.0
        saved_seconds = max(0.0, typing_seconds - max(0.0, dictation_seconds))
        return saved_seconds / 60.0

    def _save_dictionary_rules(self) -> None:
        rows = self._collect_dictionary_rows()
        ok, msg = self.app.set_dictionary_entries(rows)
        if ok:
            self.status_var.set(f"{msg} ({len(rows)} rules)")
        else:
            self.status_var.set(msg)

    def _reload_dictionary_tree(self) -> None:
        if self.dictionary_tree is None:
            return
        for row_id in self.dictionary_tree.get_children():
            self.dictionary_tree.delete(row_id)
        for source, replacement in self.app.get_dictionary_entries():
            self.dictionary_tree.insert("", "end", values=(source, replacement))

    def _copy_history(self) -> None:
        if self.history_text is None:
            return
        content = self.history_text.get("1.0", "end").strip()
        if not content:
            self.status_var.set("History is empty")
            return
        pyperclip.copy(content)
        self.status_var.set("History copied")

    def _append_history_line(self, timestamp: str, text: str) -> None:
        if self.history_text is None:
            return
        self.history_text.configure(state="normal")
        self.history_text.insert("end", f"[{timestamp}] {text}\n")
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    def _refresh_hotkeys_hint(self) -> None:
        self.hotkey_hint_var.set(
            f"Record: {self.app.config.start_stop_hotkey}\nQuit: {self.app.config.quit_hotkey}"
        )

    def _set_recording_buttons(self, listening: bool) -> None:
        self._render_home_start_icon(listening=listening)
        if self.home_toggle_caption is not None:
            self.home_toggle_caption.configure(text="Stop" if listening else "Start")
        if self.settings_toggle_btn is not None:
            self.settings_toggle_btn.configure(
                text="Stop Recording" if listening else "Start Recording",
                bg=self.c_danger if listening else self.c_success,
                activebackground=self.c_danger_hover if listening else self.c_success_hover,
            )

    def _stop_recording_from_overlay(self) -> None:
        try:
            self.app.stop_recording()
        except Exception:
            pass

    def _minimize_window(self) -> None:
        try:
            if self.custom_titlebar_enabled:
                self.root.overrideredirect(False)
            self.root.iconify()
            self.auto_minimized_on_record = True
        except Exception:
            pass

    def _restore_window_if_auto_minimized(self) -> None:
        if not self.auto_minimized_on_record:
            return
        try:
            self.root.state("normal")
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
            self.root.after(900, lambda: self.root.attributes("-topmost", False))
            if self.custom_titlebar_enabled:
                self.root.after(20, self._restore_custom_chrome)
        except Exception:
            pass
        finally:
            self.auto_minimized_on_record = False

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

        if not self.closing and not self.app.shutdown_event.is_set():
            self.root.after(60, self._poll_events)

    def _handle_event(self, event: dict) -> None:
        event_type = event.get("type")

        if event_type == "status":
            status = str(event.get("value", "idle"))
            if status == "listening":
                self.status_var.set("Listening")
                self._set_recording_buttons(True)
                self.overlay.show_listening()
                self.root.after(10, self._minimize_window)
            elif status == "processing":
                self.status_var.set("Transcribing")
                self._set_recording_buttons(False)
                self.overlay.show_processing()
                self.root.after(10, self._restore_window_if_auto_minimized)
            elif status == "idle":
                self.status_var.set("Ready")
                self._set_recording_buttons(False)
                self.overlay.hide()
                self.root.after(10, self._restore_window_if_auto_minimized)
            elif status == "stopped":
                self.status_var.set("Stopped")
                self._set_recording_buttons(False)
                self.overlay.hide()
                self.root.after(10, self._restore_window_if_auto_minimized)

        elif event_type == "level":
            self.overlay.set_level(float(event.get("value", 0.0)))

        elif event_type == "text":
            text = str(event.get("value", ""))
            self.last_text_var.set(text or "-")
            self.overlay.show_inserted(text)

        elif event_type == "stats":
            words = int(event.get("words", 0))
            seconds = float(event.get("seconds", 0.0))
            wpm = float(event.get("wpm", 0.0))
            self.words_var.set(str(words))
            self.seconds_var.set(f"{seconds / 60.0:.1f} min")
            self.time_saved_var.set(f"{self._estimate_time_saved_minutes(words, seconds):.1f} min")
            self.wpm_var.set(f"{wpm:.0f}")

        elif event_type == "history_item":
            timestamp = str(event.get("timestamp", ""))
            text = str(event.get("text", ""))
            if text:
                self._append_history_line(timestamp, text)

        elif event_type == "history_cleared":
            if self.history_text is not None:
                self.history_text.configure(state="normal")
                self.history_text.delete("1.0", "end")
                self.history_text.configure(state="disabled")
            self.status_var.set("History cleared")

        elif event_type == "dictionary_updated":
            self._reload_dictionary_tree()

        elif event_type == "hotkeys":
            self.start_hotkey_var.set(str(event.get("start", self.app.config.start_stop_hotkey)))
            self.quit_hotkey_var.set(str(event.get("quit", self.app.config.quit_hotkey)))
            self._refresh_hotkeys_hint()

        elif event_type in {"message", "error"}:
            self.status_var.set(str(event.get("text", "")))

        elif event_type == "quit":
            self._close_ui()

    def _on_close(self) -> None:
        if self.closing:
            return
        if self.close_to_tray:
            self._hide_to_tray()
            return
        self.app.stop()
        self._close_ui()

    def _close_ui(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._shutdown_tray()
        try:
            self.overlay.destroy()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lesstyper - global voice typing for Windows")
    parser.add_argument(
        "--model",
        default="small.en",
        help="Whisper model size (use multilingual models like 'small' for non-English).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (en, de, fr, es, hi, bn) or 'auto'.",
    )
    parser.add_argument("--start-stop-hotkey", default="ctrl+alt+v")
    parser.add_argument("--quit-hotkey", default="ctrl+alt+q")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-seconds", type=float, default=0.4)
    parser.add_argument("--no-trailing-space", action="store_true")
    parser.add_argument("--keep-fillers", action="store_true")
    parser.add_argument("--no-ui", action="store_true", help="Run without desktop UI")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VoiceTyperConfig:
    return VoiceTyperConfig(
        start_stop_hotkey=args.start_stop_hotkey,
        quit_hotkey=args.quit_hotkey,
        sample_rate=args.sample_rate,
        model_size=args.model,
        language=args.language,
        min_record_seconds=args.min_seconds,
        trailing_space=not args.no_trailing_space,
        remove_fillers=not args.keep_fillers,
        ui_enabled=not args.no_ui,
    )


if __name__ == "__main__":
    cli_args = parse_args()
    app = VoiceTyperApp(build_config(cli_args))
    if app.config.ui_enabled:
        VoiceTyperUI(app).run()
    else:
        app.run_headless()
