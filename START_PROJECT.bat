@echo off
setlocal
cd /d "%~dp0"

if not exist "%USERPROFILE%\.venv\Scripts\activate.bat" (
  echo Khong tim thay %USERPROFILE%\.venv
  echo Hay tao/cai moi truong Python truoc.
  pause
  exit /b 1
)

start "AI Backend" cmd /k "cd /d ""%~dp0"" && call ""%USERPROFILE%\.venv\Scripts\activate.bat"" && uvicorn api.main:app --reload"
timeout /t 4 /nobreak >nul
start "AI React" cmd /k "cd /d ""%~dp0web"" && npm run dev"
timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"
endlocal
