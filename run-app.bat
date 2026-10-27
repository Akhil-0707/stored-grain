@echo off
rem Start the capture app on port 8000, reachable from a phone on the same network.
rem Runs from this folder so models\ resolves no matter where it is launched from.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv found in %CD%. Create it first:
    echo     py -3.13 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo On this laptop:  http://localhost:8000
echo On the phone:    http://YOUR-IPv4-ADDRESS:8000  (same Wi-Fi or hotspot)
echo.
ipconfig | findstr /c:"IPv4"
echo.
echo Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" -m uvicorn src.app.server:app --host 0.0.0.0 --port 8000
pause
