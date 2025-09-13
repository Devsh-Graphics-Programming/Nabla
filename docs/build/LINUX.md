# Linux build

## Supported toolsets

- **[GCC](https://gcc.gnu.org/)**

- **[Clang](https://clang.llvm.org/)**

## Build modes

### Vanilla

Most extensions disabled.

## CMake (DEPRICATED NOTES)

Same as Windows, except that currently we have no way of setting the correct working directory for executing the examples from within the IDE (for debugging). If you care about this please submit an *issue/PR/MR* to **[*CMake's* gitlab](https://gitlab.kitware.com/cmake/cmake)**.

We recommend the ***[Codelite IDE](https://codelite.org/)*** as that has a *CMake-gui* generator and has been tested and works relatively nice.

**[*Visual Studio Code*](https://code.visualstudio.com/)** suffers from a number of issues such as configuring the *CMake* every time you want to build a target and slow build times. Here are the issues:

1. **https://github.com/microsoft/vscode-cmake-tools/issues/771**
2. **https://github.com/microsoft/vscode-cmake-tools/issues/772**
3. **https://github.com/microsoft/vscode-cmake-tools/issues/773**

***[Clang](https://clang.llvm.org/) toolset*** is unmaintained and untested on Linux.
