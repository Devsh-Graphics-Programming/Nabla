# IrrExtCEGUI
(Copyright (c) 2019 InnerPiece Technology Co., Ltd.)

This extension library provides IrrlichtBAW integration with CEGUI:
- CEGUI init/destroy and render GUI functions
- [Improved Alfisko](https://gitlab.com/InnerPieceOSS/cegui_alfisko) theme
- Mangage some common GUI features: Color Picker, File Dialog, GUI Events, Opacity, ...

To enable this extension, turn on `IRR_BUILD_CEGUI` option when configuring with CMake.
(Add `-DIRR_BUILD_CEGUI=ON` to your CMake call.)

To build IrrlichtBAW example (official projects in `example_tests` folder) with this extension, use `IRR_EXT_CEGUI_INCLUDE_DIRS` and `IRR_EXT_CEGUI_LIB` variables in your CMake code.

For end users (who build and install IrrlichtBAW with this extension enabled to a specific location, represented by `INSTALL_PATH` below), make sure that:
- `INSTALL_PATH/include/irr/ext/CEGUI/` is included in your compiler's include directories
- `INSTALL_PATH/lib/irr/ext/CEGUI/` is included in your compiler's library search paths
- For static linking, the library order matters. Refer to `3rdparty/CMakeLists.txt` from line [122](https://github.com/buildaworldnet/IrrlichtBAW/blob/master/3rdparty/CMakeLists.txt#L122) to define your linker settings.

This extension is developed and distributed under MIT license by [InnerPiece Technology Co., Ltd.](https://innerpiece.io
), Vietnam.
