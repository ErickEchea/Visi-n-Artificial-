@echo off
rem Busca el Python que tiene instalados torch/ultralytics (lo usan los demas .bat)
set "PY=C:\Users\erick\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"
