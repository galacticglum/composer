@echo off
set string=%1
set query=%2
echo.%1 | findstr /C:"%2" 1>nul && ( echo 1 ) || ( echo 0 )