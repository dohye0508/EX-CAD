@echo off
REM Builds the standalone EX-CAD.exe (no Blender required) with the Zig C++ toolchain.
REM Requires: python -m pip install ziglang
setlocal
cd /d "%~dp0"
if not exist bin mkdir bin

set CORE=..\source\blender\editors\excad
python -m ziglang c++ -target x86_64-windows-gnu -O2 -std=c++17 ^
  -I"%CORE%" ^
  main.cpp "%CORE%\excad_sketch_solver.cc" "%CORE%\excad_sdf_mesh.cc" ^
  -lopengl32 -lgdi32 -luser32 -lws2_32 -lcomdlg32 -lshell32 ^
  -Wl,--subsystem,windows ^
  -o bin\EX-CAD.exe
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo BUILD OK: %~dp0bin\EX-CAD.exe
