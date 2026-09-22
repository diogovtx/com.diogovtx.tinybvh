@echo off
rem Use the cl of an x64 developer prompt if there is one; otherwise set up the newest Visual Studio
rem with the C++ tools. A plain Developer Command Prompt targets x86, hence the check on cl's banner.
rem The parentheses matter: a pipe from a command that is not found aborts the whole batch file.
(cl 2>&1) | findstr /c:"x64" >nul || for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call "%%i\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
(cl 2>&1) | findstr /c:"x64" >nul || (echo No x64 cl.exe found: install Visual Studio with the C++ tools, or run this from an x64 Native Tools Command Prompt. & exit /b 1)
cd /d "%~dp0"
rem tiny_bvh.h is not shipped with the package: download it from the tinybvh 1.8.0 release commit.
if not exist tiny_bvh.h (
	echo Downloading tiny_bvh.h from tinybvh 1.8.0
	curl.exe -fsSL -o tiny_bvh.h https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/tiny_bvh.h || (del tiny_bvh.h 2>nul & exit /b 1)
)
cl /nologo /O2 /EHsc /std:c++20 /fp:precise refdump.cpp /Fe:refdump.exe
cl /nologo /O2 /EHsc /std:c++20 /fp:precise layoutdump.cpp /Fe:layoutdump.exe
cl /nologo /O2 /EHsc /std:c++20 /fp:precise featdump.cpp /Fe:featdump.exe
cl /nologo /O2 /EHsc /std:c++20 /fp:precise dbldump.cpp /Fe:dbldump.exe
rem NDEBUG: BVH::IntersectTLAS asserts on the BLAS layout with a list that predates VoxelSet.
cl /nologo /O2 /EHsc /std:c++20 /fp:precise /DNDEBUG voxeldump.cpp /Fe:voxeldump.exe
rem simddump needs the SIMD code paths that the other tools compile out; see its header.
rem NDEBUG: BVH::IntersectTLAS asserts on a BLAS layout list that omits BVH_SoA.
cl /nologo /O2 /EHsc /std:c++20 /fp:precise /arch:AVX2 /DNDEBUG simddump.cpp /Fe:simddump.exe
