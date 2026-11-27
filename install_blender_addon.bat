@echo off
setlocal
set SRC=%~dp0excad_addon
set DST=%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\excad_addon
if exist "%DST%" rmdir /s /q "%DST%"
xcopy /E /I /Y "%SRC%" "%DST%" >nul
if errorlevel 1 (
  echo INSTALL FAILED
  exit /b 1
)
echo INSTALLED: %DST%
