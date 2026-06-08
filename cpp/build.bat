@echo off
setlocal

set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
    echo Could not find vcvars64.bat. Install Visual Studio 2022 with the C++ workload.
    exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

if not exist build (
    cmake -B build -G Ninja -DGOLDY_BUILD_FROM_SOURCE=ON -DGOLDY_BUILD_EXAMPLES=ON
    if errorlevel 1 exit /b 1
)

cmake --build build --target triangle %*
exit /b %ERRORLEVEL%
