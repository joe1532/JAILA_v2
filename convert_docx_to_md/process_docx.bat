@echo off
setlocal enabledelayedexpansion

REM --- Konfiguration ---
REM Stierne er nu relative til placeringen af denne batch-fil.
set SCRIPT_PATH=%~dp0docx_to_md_preprocessor.py
set INPUT_DIR=%~dp0input preprocess
set OUTPUT_DIR=%~dp0input markdown
REM ---------------------

echo.
echo ================================================
echo         DOCX til Markdown Konverter
echo ================================================
echo.

REM Tjek om Python-scriptet findes
if not exist "%SCRIPT_PATH%" (
    echo Fejl: Python-scriptet blev ikke fundet ved: "%SCRIPT_PATH%"
    echo Kontroller, at docx_to_md_preprocessor.py ligger i samme mappe som denne batch-fil.
    echo.
    pause
    goto :eof
)

REM Opret input/output mapper i denne mappe (convert_docx_to_md), hvis de ikke findes
if not exist "%INPUT_DIR%" (
    echo Opretter input preprocess-mappe: "%INPUT_DIR%"
    mkdir "%INPUT_DIR%"
)
if not exist "%OUTPUT_DIR%" (
    echo Opretter input markdown-mappe: "%OUTPUT_DIR%"
    mkdir "%OUTPUT_DIR%"
)

REM Tjek om der er .docx-filer i input-mappen
dir /b "%INPUT_DIR%\*.docx" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Ingen .docx-filer fundet i mappen: "%INPUT_DIR%"
    echo Læg venligst dine .docx-filer der, og kør scriptet igen.
    echo.
    pause
    goto :eof
)

echo Starter konvertering af .docx-filer fra '%INPUT_DIR%'...
echo Output vil blive gemt i '%OUTPUT_DIR%'...
echo.

REM Kør Python-scriptet med validering
python "%SCRIPT_PATH%" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --validate --verbose

echo.
echo ================================================
echo              Konvertering færdig!
echo ================================================
echo.
echo Markdown-filer er gemt i mappen: "%OUTPUT_DIR%"
echo.
pause
