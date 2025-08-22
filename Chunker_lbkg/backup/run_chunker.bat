@echo off
setlocal

:: =================================================================
:: == JAILA Chunker - Robust Batch Runner
:: == Formål: Kører Python-scriptet og fanger alt output i en logfil
:: =================================================================

:: Naviger til mappen hvor batch-filen ligger
cd /d "%~dp0"

:: Definer stier og filnavne
set SCRIPT_NAME=chunkerlbkg_debug.py
set INPUT_FILE="input\Kildeskatteloven (2025-04-11 nr. 460).md"
set OUTPUT_PREFIX="output\final_output"
set LOG_FILE="debug_output.log"
set MAX_TOKENS=275

:: Ryd gammel logfil for at starte friskt
if exist %LOG_FILE% del %LOG_FILE%

echo.
echo ============================================================ >> %LOG_FILE%
echo ==         Starter JAILA Chunker - %TIME%         == >> %LOG_FILE%
echo ============================================================ >> %LOG_FILE%
echo. >> %LOG_FILE%

echo Kører nu scriptet. Alt output (både succes og fejl) bliver gemt i:
echo %LOG_FILE%
echo.
echo Vent venligst...
echo.

:: Tjek om Python-scriptet eksisterer
if not exist "%SCRIPT_NAME%" (
    echo [FEJL] Scriptet %SCRIPT_NAME% blev ikke fundet! >> %LOG_FILE%
    echo Scriptet %SCRIPT_NAME% blev ikke fundet! Tjek filnavnet.
    goto :end
)

:: Tjek om input-filen eksisterer
if not exist %INPUT_FILE% (
    echo [FEJL] Input-filen %INPUT_FILE% blev ikke fundet! >> %LOG_FILE%
    echo Input-filen %INPUT_FILE% blev ikke fundet! Tjek stien.
    goto :end
)

:: Byg kommandoen
set COMMAND=python -B "%SCRIPT_NAME%" --input %INPUT_FILE% --max-tokens %MAX_TOKENS% --out-prefix %OUTPUT_PREFIX%

echo Kommando der køres: >> %LOG_FILE%
echo %COMMAND% >> %LOG_FILE%
echo. >> %LOG_FILE%
echo --- SCRIPT OUTPUT STARTER HER --- >> %LOG_FILE%
echo. >> %LOG_FILE%

:: Kør kommandoen og omdiriger BÅDE standard output OG fejl output til logfilen
%COMMAND% >> %LOG_FILE% 2>&1

:: Tjek om scriptet kørte succesfuldt
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================
    echo == SCRIPT KORTE SUCCESFULDT ✅ ==
    echo =================================
    echo.
    echo Se output-filerne i 'output' mappen.
    echo Detaljeret log er gemt i %LOG_FILE%
) else (
    echo.
    echo =================================
    echo ==    SCRIPT FAILEDT ❌       ==
    echo =================================
    echo.
    echo En fejl opstod under kørslen (Fejlkode: %ERRORLEVEL%).
    echo Se den fulde fejlmeddelelse i: %LOG_FILE%
)

:end
echo.
echo Processen er færdig.
pause
