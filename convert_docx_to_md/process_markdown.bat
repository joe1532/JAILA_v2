@echo off
setlocal enabledelayedexpansion

REM --- Konfiguration ---
REM Stierne er nu relative til placeringen af denne batch-fil.
set SCRIPT_PATH=%~dp0add_markdown_markers.py
set INPUT_DIR=%~dp0input markdown
set CHUNKER_INPUT_DIR=%~dp0..\\Chunker_lbkg\\input
set READY_FOR_CHUNK_DIR=%~dp0ready for chunk
REM ---------------------

echo.
echo ================================================
echo       Markdown Markers Tilføjer
echo ================================================
echo.

REM Tjek om Python-scriptet findes
if not exist "%SCRIPT_PATH%" (
    echo Fejl: Python-scriptet blev ikke fundet ved: "%SCRIPT_PATH%"
    echo Kontroller, at add_markdown_markers.py ligger i samme mappe som denne batch-fil.
    echo.
    pause
    goto :eof
)

REM Opret input/output mapper, hvis de ikke findes
if not exist "%INPUT_DIR%" (
    echo Opretter input markdown-mappe: "%INPUT_DIR%"
    mkdir "%INPUT_DIR%"
)
if not exist "%CHUNKER_INPUT_DIR%" (
    echo Opretter Chunker_lbkg input-mappe: "%CHUNKER_INPUT_DIR%"
    mkdir "%CHUNKER_INPUT_DIR%"
)
if not exist "%READY_FOR_CHUNK_DIR%" (
    echo Opretter ready for chunk-mappe: "%READY_FOR_CHUNK_DIR%"
    mkdir "%READY_FOR_CHUNK_DIR%"
)

REM Tjek om der er .md-filer i input-mappen
dir /b "%INPUT_DIR%\\*.md" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Ingen .md-filer fundet i mappen: "%INPUT_DIR%"
    echo Læg venligst dine .md-filer der, og kør scriptet igen.
    echo.
    pause
    goto :eof
)

echo Starter tilføjelse af markdown-markører fra '%INPUT_DIR%'...
echo Primært output vil blive gemt i '%CHUNKER_INPUT_DIR%'...
echo.

REM Kør Python-scriptet for at placere filerne i Chunker_lbkg/input
python "%SCRIPT_PATH%" --input "%INPUT_DIR%" --output "%CHUNKER_INPUT_DIR%" --overwrite

echo.
echo Kopierer de færdige filer til '%READY_FOR_CHUNK_DIR%'...
copy "%CHUNKER_INPUT_DIR%\\*.md" "%READY_FOR_CHUNK_DIR%\\"

echo.
echo ================================================
echo         Markdown-markører tilføjet!
echo ================================================
echo.
echo Færdige filer er gemt i:
echo   - %CHUNKER_INPUT_DIR%
echo   - %READY_FOR_CHUNK_DIR%
echo.
pause
