@echo off
REM Checks whether conda is installed. Outputs 1 if it is; otherwise 0.
conda --version >nul 2>&1 && ( echo 1 ) || ( echo 0 )