@echo off

setlocal
for /f "tokens=* USEBACKQ" %%f in (`%~dp0/utils/check_conda.bat`) do ( set __composervar_HAS_CONDA=%%f )
if %__composervar_HAS_CONDA% == 0 (
    call:print_error "- conda not detected! Composer requires conda."
    exit /B 1
)
endlocal

echo - conda detected
echo - setting up composer environment

setlocal ENABLEDELAYEDEXPANSION
for /f "tokens=* USEBACKQ" %%f in (`%~dp0/utils/conda_env_exists.bat composer`) do ( set __composervar_COMPOSER_ENV_EXISTS=%%f )
if %__composervar_COMPOSER_ENV_EXISTS% == 1 (
    set /P PROMPT_RESPONSE="conda environment of name 'composer' already exists! Would you like to remove this environment and continue (Y/[N])? "
    if /I !PROMPT_RESPONSE! neq Y (
        exit /B 0
    )

    echo - removing old conda environment
    for /f "tokens=* USEBACKQ" %%f in (`%~dp0/utils/current_conda_env.bat`) do (
        set env_name=%%f
        if !env_name! == "composer" (
            call conda deactivate
        )
    )

    call conda remove --name composer --all -y -q
)
endlocal

echo - creating composer conda environment
call conda env create -f environment.yml -q
call conda activate composer

setlocal ENABLEDELAYEDEXPANSION
for /f "tokens=* USEBACKQ" %%f in (`%~dp0/utils/current_conda_env.bat`) do (
    set env_name=%%f
    if not !env_name! == "composer" (
        call:print_error "Did not successfully activate the composer conda environment"
        exit /B 1
    )
)
endlocal

echo - installing composer module via pip
call pip install --editable .

echo.
echo ==============================
echo Composer Install Success!
echo.
echo NOTE:
echo For changes to become active, you will need to open a new terminal.
echo.
echo To uninstall the environment run:
echo    conda remove -n composer --all
echo.
echo To run composer, activate your environment:
echo    source activate composer
echo.
echo You can deactivate when you're done:
echo    source deactivate
echo ==============================
echo.

exit /B 0

:print_error
    set mydate=%date:~10,4%%date:~6,2%/%date:~4,2%
    set mytime=%time%
    echo %mydate%T%mytime%: %~1
    echo.
    echo ==================================================
    echo Installation did not finish successfully.
    echo For assistance, post an issue on the project page.
    echo https://github.com/galacticglum/composer
    echo ==================================================
    echo.
goto:eof