@echo off
rem Usage: run_all.bat [TestData folder]
rem The folder holds the three scenes and receives the dumps. It defaults to the TestData folder of
rem the Unity project the package is embedded in, four levels above this one.
setlocal
set "TESTDATA=%~f1"
if not defined TESTDATA for %%i in ("%~dp0..\..\..\..\TestData") do set "TESTDATA=%%~fi"
cd /d "%~dp0"
"%~dp0refdump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.ref"
"%~dp0refdump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.ref"
"%~dp0refdump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.ref"
"%~dp0layoutdump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.layouts.ref"
"%~dp0layoutdump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.layouts.ref"
"%~dp0layoutdump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.layouts.ref"
"%~dp0featdump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.feat.ref"
"%~dp0featdump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.feat.ref"
"%~dp0featdump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.feat.ref"
"%~dp0dbldump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.dbl.ref"
"%~dp0dbldump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.dbl.ref"
"%~dp0dbldump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.dbl.ref"
"%~dp0voxeldump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.vox.ref"
"%~dp0voxeldump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.vox.ref"
"%~dp0voxeldump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.vox.ref"
"%~dp0simddump.exe" "%TESTDATA%\bunny.bin"        "%TESTDATA%\bunny.simd.ref"
"%~dp0simddump.exe" "%TESTDATA%\suzanne.bin"      "%TESTDATA%\suzanne.simd.ref"
"%~dp0simddump.exe" "%TESTDATA%\cryteksponza.bin" "%TESTDATA%\cryteksponza.simd.ref"
endlocal
