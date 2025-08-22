@echo off
setlocal enabledelayedexpansion

REM --- Test Atom_ID Implementation (ASCII-Safe + Kind + Coordinates) ---
REM Kører den nye chunkerlbkg.py med alle atom_id ændringer
REM 
REM NYE FUNKTIONER:
REM - Kind inkluderet i atom_id: --kindkind (undgår kollisioner)
REM - ASCII-sikker paragraf: 'par' i stedet for '§'
REM - Note-koordinater: --parX--stkY--kindnote--noteN
REM - atom_base_id: Stabil base-ID uden part-suffix
REM - Unikheds-QA: Fejler ved dublerede atom_id'er
REM
REM --- Konfiguration ---
set SCRIPT_PATH=%~dp0chunkerlbkg.py
set INPUT_DIR=%~dp0..\input
set OUTPUT_DIR=%~dp0output
REM ---------------------

REM Tjek om Python-scriptet findes
if not exist "%SCRIPT_PATH%" (
    echo Fejl: Python-scriptet blev ikke fundet ved: "%SCRIPT_PATH%"
    echo Kontroller, at chunkerlbkg.py ligger i test-mappen.
    pause
    goto :eof
)

REM Opret output-mappe hvis den ikke findes
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
echo ========================================================
echo TESTING ATOM_ID IMPLEMENTATION
echo ========================================================
echo Script: %SCRIPT_PATH%
echo Input:  %INPUT_DIR%
echo Output: %OUTPUT_DIR%
echo.
echo NYE FUNKTIONER I DENNE VERSION:
echo - ASCII-sikre atom_id'er: bruger 'par' i stedet for '§'
echo - Kind inkluderet i ID: --kindkind (undgår kollisioner)
echo - Note-koordinater: --parX--stkY--kindnote--noteN
echo - atom_base_id: Stabil base uden part-suffix
echo - Unikheds-QA: Fejler ved dublerede ID'er
echo ========================================================
echo.

REM Loop gennem alle .md-filer i input-mappen
for %%f in ("%INPUT_DIR%\*.md") do (
    echo --------------------------------------------------
    echo Behandler fil: "%%f"

    REM Udtræk filnavnet uden sti og endelse
    set "FILENAME=%%~nf"

    REM Definer output-sti
    set "OUTPUT_PREFIX=%OUTPUT_DIR%\!FILENAME!"

    REM Kør Python-scriptet med alle nye funktioner
    echo.
    echo DEBUG: Kører kommando:
    echo python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!" --no-csv --max-tokens 275
    echo.
    
    REM Kør med fejlfangst
    python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!" --no-csv --max-tokens 275
    
    if errorlevel 1 (
        echo.
        echo ❌ FEJL: Script fejlede! Dette kan være:
        echo    - Dublerede atom_id'er ^(unikheds-QA fejl^)
        echo    - Parsing-fejl
        echo    - Input-format problemer
        echo.
        echo Tjek output ovenfor for detaljer.
        pause
        goto :eof
    ) else (
        echo.
        echo ✅ SUCCESS: Fil behandlet uden fejl!
        echo.
    )
)

echo --------------------------------------------------
echo.
echo 🎉 ALLE FILER BEHANDLET SUCCESFULDT!
echo.
echo Output-filer er gemt i mappen: "%OUTPUT_DIR%"
echo.
echo GENEREREDE FILER:
echo - *_chunks.jsonl  ^(JSON Lines format^)
echo - *_chunks.json   ^(Standard JSON array^)
echo.
echo NYE FELTER I OUTPUT:
echo - atom_id: ASCII-sikker deterministisk ID ^(par1--stk1--nr3--kindrule^)
echo - atom_base_id: Stabil base-ID uden part-suffix
echo - kind: Atomtype ^(rule/note/section/parent^)
echo.
echo TJEK DISSE FORBEDRINGER:
echo 1. Ingen '§' tegn i atom_id ^(ASCII-sikker^)
echo 2. Kind inkluderet i alle ID'er ^(undgår kollisioner^)
echo 3. Noter har koordinat-kontekst når muligt
echo 4. Ingen dublerede atom_id'er ^(QA tjekket^)
echo.
pause
