@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_python.bat"
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║        VISION IA - Selecciona modo       ║
echo  ╚══════════════════════════════════════════╝
echo.
echo   [1] Detectar con CAMARA en tiempo real
echo   [2] Detectar en VIDEO (guarda el resultado en \resultados)
echo   [3] VIDEO rapido (sin TTA)
echo   [4] Debug del modelo Roboflow (foto de la camara)
echo.
set /p opcion="Ingresa 1, 2, 3 o 4: "

if "%opcion%"=="1" (
    "%PY%" detectar_camara.py
) else if "%opcion%"=="2" (
    "%PY%" detectar_video.py
) else if "%opcion%"=="3" (
    "%PY%" detectar_video.py --sin-tta
) else if "%opcion%"=="4" (
    "%PY%" debug_modelo.py
) else (
    echo Opcion no valida.
)
pause
