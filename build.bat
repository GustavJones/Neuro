@echo off

:: CMake Build Type
set BUILD_TYPE=Debug

:: Set current directory to script directory
cd /d "%~dp0"

:: Configure cmake project
cmake.exe -S . -B build -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

:: Build cmake project
cmake --build build --config %BUILD_TYPE% -j
