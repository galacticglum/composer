@echo off
setlocal ENABLEDELAYEDEXPANSION
set count=1
set env_name=%1
for /f "tokens=* USEBACKQ" %%f in (`conda info --envs`) do (
    for /f "tokens=* USEBACKQ" %%i in (`%~dp0/contains_string.bat "%%f" "!env_name!"`) do (
        if "%%i" == "1 " (
            echo 1
            exit /B 0
        )
    )
    set /a count=!count!+1
)
echo 0
endlocal