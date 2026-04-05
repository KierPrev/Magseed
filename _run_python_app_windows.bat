@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "APP_SCRIPT=%~1"
set "APP_NAME=%~2"

if "%APP_SCRIPT%"=="" (
    echo Uso: _run_python_app_windows.bat archivo.py "Nombre App"
    pause
    exit /b 1
)

if "%APP_NAME%"=="" set "APP_NAME=%APP_SCRIPT%"

call :select_python
if errorlevel 1 (
    pause
    exit /b 1
)

if not exist ".venv-win\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual para %APP_NAME%...
    %PYTHON_CMD% -m venv ".venv-win"
    if errorlevel 1 (
        echo.
        echo No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

echo [2/3] Instalando dependencias...
call ".venv-win\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

python -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 (
    echo.
    echo No se pudo actualizar pip.
    pause
    exit /b 1
)

python -m pip install --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 (
    echo.
    echo No se pudieron instalar las dependencias.
    pause
    exit /b 1
)

echo [3/3] Abriendo %APP_NAME%...
cd /d "%~dp0src"
python "%APP_SCRIPT%"
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
