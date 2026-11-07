@echo off
title Lepidoptera AI - Butterfly vs Moth Classifier
echo ========================================================
echo  Launching Lepidoptera AI Classification Web App...
echo ========================================================
echo.
start "" "http://127.0.0.1:5000"
py -3.12 app.py
if %ERRORLEVEL% NEQ 0 (
    echo Falling back to default python...
    python app.py
)
pause
