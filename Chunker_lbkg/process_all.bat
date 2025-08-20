@echo off
setlocal enabledelayedexpansion

REM --- Konfiguration ---
REM Stierne er nu relative til placeringen af denne batch-fil.
set SCRIPT_PATH=%~dp0chunkerlbkg.py
set INPUT_DIR=%~dp0input
set OUTPUT_DIR=%~dp0output
REM ---------------------

REM Tjek om Python-scriptet findes
if not exist "%SCRIPT_PATH%" (
    echo Fejl: Python-scriptet blev ikke fundet ved: "%SCRIPT_PATH%"
    echo Kontroller, at chunkerlbkg.py ligger i samme mappe som denne batch-fil.
    pause
    goto :eof
)

REM Opret input/output mapper i denne mappe (Chunker_lbkg), hvis de ikke findes
if not exist "%INPUT_DIR%" (
    echo Opretter input-mappe: "%INPUT_DIR%"
    mkdir "%INPUT_DIR%"
)
if not exist "%OUTPUT_DIR%" (
    echo Opretter output-mappe: "%OUTPUT_DIR%"
    mkdir "%OUTPUT_DIR%"
)

REM Tjek om der er .md-filer i input-mappen
dir /b "%INPUT_DIR%\*.md" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Ingen .md-filer fundet i mappen: "%INPUT_DIR%"
    echo Læg venligst dine .md-filer der, og kør scriptet igen.
    echo.
    pause
    goto :eof
)

echo.
echo Starter behandling af .md-filer fra '%INPUT_DIR%'...
echo.

REM Loop gennem alle .md-filer i input-mappen
for %%f in ("%INPUT_DIR%\*.md") do (
    echo --------------------------------------------------
    echo Behandler fil: "%%f"

    REM Udtræk filnavnet uden sti og endelse
    set "FILENAME=%%~nf"

    REM Definer output-sti
    set "OUTPUT_PREFIX=%OUTPUT_DIR%\!FILENAME!"

    REM Kør Python-scriptet
    python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!"
    
    echo.
)

echo --------------------------------------------------
echo.
echo Alle filer er blevet behandlet.
echo Output-filer er gemt i mappen: "%OUTPUT_DIR%"
echo.
pause
