:: Nabla CI Pipeline Framework Module proxy

@echo off
@setlocal

:: Get the directory where this platform proxy script is located
set scriptDirectory=%~dp0

:: Change the current working directory to the cross-platform Python build script
cd /d "%scriptDirectory%"

:: Execute implementation of the pipeline build script in Python
python -m %*