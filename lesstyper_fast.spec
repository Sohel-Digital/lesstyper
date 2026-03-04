# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = []
binaries = []
hiddenimports = ["fontawesomefree"]

datas += collect_data_files("faster_whisper", includes=["assets/*"])
datas += collect_data_files(
    "fontawesomefree",
    includes=[
        "static/fontawesomefree/webfonts/fa-solid-900.ttf",
        "static/fontawesomefree/js-packages/@fortawesome/fontawesome-free/webfonts/fa-solid-900.ttf",
    ],
)

for pkg_name in ("sounddevice", "ctranslate2", "av"):
    try:
        binaries += collect_dynamic_libs(pkg_name)
    except Exception:
        pass

excludes = ["onnxruntime", "tensorflow", "torch", "torchvision", "torchaudio"]

a = Analysis(
    ["voice_type.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lesstyper_fast",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lesstyper_fast",
)
