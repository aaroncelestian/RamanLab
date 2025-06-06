# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['raman_analysis_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['version'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RamanLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='RamanLab_icon.ico',  # Windows icon
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RamanLab',
)

# macOS app bundle
app = BUNDLE(
    coll,
    name='RamanLab.app',
    icon='RamanLab_icon.icns',  # macOS icon
    bundle_identifier='org.ramanlab.RamanLab',
    info_plist={
        'CFBundleName': 'RamanLab',
        'CFBundleDisplayName': 'RamanLab',
        'CFBundleIdentifier': 'org.ramanlab.RamanLab',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleInfoDictionaryVersion': '6.0',
        'NSHighResolutionCapable': True,
        'LSApplicationCategoryType': 'public.app-category.productivity',
    },
)
