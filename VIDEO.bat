@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_python.bat"
rem Puedes arrastrar un video encima de este archivo para procesarlo
"%PY%" detectar_video.py %*
pause
