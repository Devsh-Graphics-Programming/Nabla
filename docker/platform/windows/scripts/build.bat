@echo off
@setlocal

:: Get the directory where this platform proxy script is located
set scriptDirectory=%~dp0

:: Change the current working directory to the cross-platform Python build script
cd /d "%scriptDirectory%\..\..\..\scripts"

:: Execute implementation of the pipeline build script in Python
python ./build.py %*