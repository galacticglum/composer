@echo off
setlocal ENABLEDELAYEDEXPANSION
set count=1
for /f "tokens=* USEBACKQ" %%f in (`conda info --envs`) do (
    for /f "tokens=* USEBACKQ" %%i in (`%~dp0/contains_string.bat "%%f" *`) do (
        if "%%i" == "1 " (
            for /f "tokens=1 delims= " %%j in ("%%f") do (
                set result=%%j
                echo "!result!"
                exit /B 0
            )
        )
    )
    set /a count=!count!+1
)
endlocal