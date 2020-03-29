@setlocal enableextensions enabledelayedexpansion
@echo off
set str1="bcd"
if not x%str1:bcd=%==x%str1% echo It contains bcd
endlocal