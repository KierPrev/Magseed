@echo off
setlocal EnableExtensions

cd /d "%~dp0"

call :select_python
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

echo [1/3] Actualizando pip...
%PYTHON_CMD% -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 (
    echo.
    echo No se pudo actualizar pip.
    pause
    exit /b 1
)

echo [2/3] Instalando dependencias...
%PYTHON_CMD% -m pip install --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 (
    echo.
    echo No se pudieron instalar las dependencias.
    pause
    exit /b 1
)

echo [3/3] Abriendo PipelineGUI...
cd /d "%~dp0src"
%PYTHON_CMD% "PipelineGUI.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo La aplicacion termino con error %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%

:select_python
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    exit /b 0
)

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    exit /b 0
)

py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    exit /b 0
)

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)

echo No se encontro Python instalado.
echo Instala Python 3 para Windows desde https://www.python.org/downloads/windows/
exit /b 1
