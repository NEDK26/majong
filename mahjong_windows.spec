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

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    analysis.binaries,
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
    # 自动化自检遇到启动错误时直接返回非零，不弹出会挂住构建的异常对话框。
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
