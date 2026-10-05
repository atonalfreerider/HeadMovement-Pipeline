@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%~dp0..\.venv-nopo4d\Scripts;%PATH%"
set TORCH_CUDA_ARCH_LIST=7.5
set MAX_JOBS=4
set PYTHONUTF8=1
set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
"%~dp0..\.venv-nopo4d\Scripts\python.exe" %*
