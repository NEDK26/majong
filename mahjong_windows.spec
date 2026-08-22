# -*- mode: python ; coding: utf-8 -*-

"""精简的 Windows 单文件构建配置。"""

analysis = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("assets/ocr_templates", "assets/ocr_templates")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PIL",
        "asyncio",
        "email",
        "http",
        "multiprocessing",
        "pydoc",
        "ssl",
        "_ssl",
        "urllib",
        "xml",
    ],
    noarchive=False,
    optimize=1,
)

# opencv-python-headless 仍附带约 30 MB 的视频编解码器；本程序只处理静态帧，
# 不打开或写入视频，因此可以安全移除。
runtime_binaries = [
    item
    for item in analysis.binaries
    if "opencv_videoio_ffmpeg" not in item[0].lower()
]

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    runtime_binaries,
    analysis.datas,
    [],
    name="MahjongStudyAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
