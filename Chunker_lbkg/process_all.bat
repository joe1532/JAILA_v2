@echo off
setlocal enabledelayedexpansion

REM --- Chunker Batch Script ---
REM Kører chunkerlbkg.py på alle .md-filer i input-mappen
REM 
REM SPLIT-FUNKTIONALITET:
REM - Standard: 275 tokens per chunk (aktiveret)
REM - For at deaktivere: ændr --max-tokens 275 til --max-tokens 0
REM - For anden grænse: ændr til fx --max-tokens 400
REM
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

    REM Kør Python-scriptet med split-funktionalitet aktiveret (275 tokens)
    REM For at deaktivere split: tilføj --max-tokens 0
    python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!" --no-csv --max-tokens 275
    
    echo.
)

echo --------------------------------------------------
echo.
echo Alle filer er blevet behandlet.
echo Output-filer er gemt i mappen: "%OUTPUT_DIR%"
echo.
pause
