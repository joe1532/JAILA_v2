@echo off
setlocal enabledelayedexpansion

REM --- Chunker Batch Script (Phase 3 - ASCII-Safe Atom_ID) ---
REM Kører chunkerlbkg.py på alle .md-filer i input-mappen
REM 
REM NYE FUNKTIONER I PHASE 3:
REM - ASCII-sikre atom_id'er: bruger 'par' i stedet for '§'
REM - Stabile base_id'er: atom_base_id uden part-suffix for grouping
REM - Auto law_id afledning: fra filnavn (fx KSL_2025-04-11_nr460)
REM - Udvidede QA-tjek: validerer konsistens og unikhed
REM - Litra/punkt support: fuld hierarki-parsing
REM - JSON + JSONL output: begge formater genereres
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

    REM Kør Python-scriptet med alle nye funktioner aktiveret
    REM --max-tokens 275: Split-funktionalitet (0 = deaktiveret)
    REM --law-id: Kan angives manuelt (ellers auto-afledt fra filnavn)
    REM --no-csv: Undgår CSV-output (kun JSON/JSONL)
    echo.
    echo DEBUG: Kører kommando:
    echo python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!" --no-csv --max-tokens 275
    echo.
    python "%SCRIPT_PATH%" --input "%%f" --out-prefix "!OUTPUT_PREFIX!" --no-csv --max-tokens 275
    
    echo.
)

echo --------------------------------------------------
echo.
echo Alle filer er blevet behandlet.
echo Output-filer er gemt i mappen: "%OUTPUT_DIR%"
echo.
echo GENEREREDE FILER:
echo - *_chunks.jsonl  (JSON Lines format)
echo - *_chunks.json   (Standard JSON array)
echo.
echo NYE FELTER I OUTPUT:
echo - atom_id: ASCII-sikker deterministisk ID (par1--stk1--nr3--kindrule)
echo - atom_base_id: Stabil base-ID uden part-suffix
echo - kind: Atomtype (rule/note/section/parent)
echo - law_id: Auto-afledt fra filnavn
echo.
pause
