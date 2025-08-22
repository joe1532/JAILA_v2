@echo off
setlocal

REM Gå til projektroden (én mappe op fra denne BAT)
pushd "%~dp0\.."

REM Konfiguration
set "SCRIPT_PATH=Chunker_lbkg\chunkerlbkg  (bu3).py"
set "DEFAULT_INPUT=input\Kildeskatteloven (2025-04-11 nr. 460).md"
set "ALT_INPUT=Chunker_lbkg\input\Kildeskatteloven (2025-04-11 nr. 460).md"
set "OUTPUT_DIR=Chunker_lbkg\output"
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)
set "LOG_FILE=Chunker_lbkg\debug_output_bu3.log"

REM Vælg input (1) fra parameter, (2) default, (3) alternativ placering
set "INPUT_FILE=%DEFAULT_INPUT%"
if not "%~1"=="" (
    set "INPUT_FILE=%~1"
)
if not exist "%INPUT_FILE%" (
    if exist "%ALT_INPUT%" (
        set "INPUT_FILE=%ALT_INPUT%"
    )
)

REM Udled base filnavn for out-prefix
for %%F in ("%INPUT_FILE%") do set "BASENAME=%%~nF"
set "OUTPUT_PREFIX=%OUTPUT_DIR%\%BASENAME%"

REM Ryd logfil for en ren start
echo ========================================================= > "%LOG_FILE%"
echo            Kører chunkerlbkg (bu3).py                     >> "%LOG_FILE%"
echo ========================================================= >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [INFO] Arbejdsmappe: %CD% >> "%LOG_FILE%"
echo [INFO] Script: "%SCRIPT_PATH%" >> "%LOG_FILE%"
echo [INFO] Input:  "%INPUT_FILE%" >> "%LOG_FILE%"
echo [INFO] Output: "%OUTPUT_PREFIX%" >> "%LOG_FILE%"

REM Tjek input findes
if not exist "%INPUT_FILE%" (
    echo [FEJL] Inputfil ikke fundet: "%INPUT_FILE%" >> "%LOG_FILE%"
    echo [FEJL] Prøvede også: "%DEFAULT_INPUT%" og "%ALT_INPUT%" >> "%LOG_FILE%"
    echo [TIP] Angiv sti som parameter til BAT:  run_bu3_chunker.bat "fuld\sti\fil.md" >> "%LOG_FILE%"
    echo.
    echo Tryk en tast for at lukke...
    pause >nul
    popd
    exit /b 1
)

REM Kør scriptet med live-stream til konsol og samtidig logning
echo Starter script...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; & python -u -B \"%SCRIPT_PATH%\" --input \"%INPUT_FILE%\" --out-prefix \"%OUTPUT_PREFIX%\" --max-tokens 275 --no-csv 2>&1 | Tee-Object -FilePath \"%LOG_FILE%\" -Append"

REM Tjek om scriptet fejlede
if %errorlevel% neq 0 (
    echo.
    echo ##############################################
    echo #                                            #
    echo #      ! ! !   S C R I P T   F E J L E D E   ! ! !      #
    echo #                                            #
    echo #   Se logfilen for detaljer:                #
    echo #   %LOG_FILE%                 #
    echo #                                            #
    echo ##############################################
    echo.
    echo Tryk en tast for at lukke...
    pause >nul
    popd
    exit /b %errorlevel%
)

echo.
echo ##############################################
echo #                                            #
echo #      S C R I P T   F E R D I G T           #
echo #                                            #
@echo #   %OUTPUT_PREFIX%_chunks.jsonl             #
@echo #   (mappe: %OUTPUT_DIR%)                    #
echo #   Logfil findes her:                       #
@echo #   %LOG_FILE%                               #
echo #                                            #
echo ##############################################

echo.
echo Tryk en tast for at lukke...
pause >nul

popd
endlocal
