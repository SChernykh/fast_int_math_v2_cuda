@echo off
call "%VS140COMNTOOLS%/../../VC/vcvarsall.bat" amd64
nvcc -ptx -o kernel.ptx kernel.cu
