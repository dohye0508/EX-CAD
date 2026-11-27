@echo off
REM Launch the standalone C++ EX-CAD (no Blender). Builds it first if missing.
if not exist "%~dp0standalone\bin\EX-CAD.exe" call "%~dp0standalone\build.bat" || exit /b 1
start "" "%~dp0standalone\bin\EX-CAD.exe" %*
