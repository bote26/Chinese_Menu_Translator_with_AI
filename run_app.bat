@echo off
setlocal EnableExtensions
title Menu Vision
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  Menu Vision launcher.
rem
rem  The app targets the CURRENT Streamlit API (st.image(width="stretch"),
rem  st.metric(border=True), ...). The system Python310 has Streamlit 1.39,
rem  which predates all of that and throws
rem      TypeError: '<=' not supported between instances of 'str' and 'int'
rem  So this script keeps a dedicated .venv next to the project that carries a
rem  modern Streamlit, while inheriting the already-installed pandas / pydantic /
rem  requests from the system interpreter instead of reinstalling them.
rem
rem  Usage:   run_app.bat            -> http://localhost:8501
rem           run_app.bat 8502       -> http://localhost:8502
rem ---------------------------------------------------------------------------

set "PROJECT=%~dp0"
set "BOOTSTRAP_PY=C:\Users\eduar\AppData\Local\Programs\Python\Python310\python.exe"
set "VENV_PY=%PROJECT%.venv\Scripts\python.exe"
set "MIRROR=https://mirrors.aliyun.com/pypi/simple/"
set "PORT=8501"
if not "%~1"=="" set "PORT=%~1"

rem ---------------------------------------------------------------------------
rem  Drop inherited proxy variables.
rem
rem  These are set by whatever launched this window - a VPN client, an IDE agent,
rem  a terminal from some other tool. They outlive the proxy they point at, and
rem  then every model call dies with
rem      ProxyError: Cannot connect to proxy
rem  which looks like the API being down. Clearing them is safe: requests still
rem  honours the Windows system proxy setting, so Clash/V2Ray on 127.0.0.1:7890
rem  keeps working. To pin a specific proxy, put it in .env instead:
rem      MENU_LLM_PROXY=http://127.0.0.1:7890
rem      MENU_LLM_PROXY=off        (always direct)
rem ---------------------------------------------------------------------------
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="

echo ============================================================
echo  Menu Vision
echo    project : %PROJECT%
echo    port    : %PORT%
echo ============================================================
echo.

if not exist "%VENV_PY%" goto :bootstrap
goto :check_streamlit

rem ---------------------------------------------------------------------------
:bootstrap
echo [setup] No .venv found - creating it (one time, about a minute)...
if not exist "%BOOTSTRAP_PY%" (
    echo [error] Base interpreter not found:
    echo         %BOOTSTRAP_PY%
    echo         Fix BOOTSTRAP_PY at the top of this .bat.
    goto :fail
)
"%BOOTSTRAP_PY%" -m venv --system-site-packages "%PROJECT%.venv"
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -U pip --index-url %MIRROR% --trusted-host mirrors.aliyun.com --timeout 60
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -U streamlit --index-url %MIRROR% --trusted-host mirrors.aliyun.com --timeout 60
if errorlevel 1 goto :fail

rem ---------------------------------------------------------------------------
:check_streamlit
rem Feature probe, not a version string: does this Streamlit know the arguments
rem the app actually passes?
"%VENV_PY%" -c "import inspect, streamlit as st, sys; a = inspect.signature(st.image).parameters; b = inspect.signature(st.metric).parameters; sys.exit(0 if ('width' in a and 'border' in b) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [setup] Streamlit in .venv is too old for this app - upgrading...
    "%VENV_PY%" -m pip install -U streamlit --index-url %MIRROR% --trusted-host mirrors.aliyun.com --timeout 60
    if errorlevel 1 goto :fail
)

"%VENV_PY%" -c "import sys, streamlit; print('   python    :', sys.version.split()[0]); print('   streamlit :', streamlit.__version__)"
echo.

rem ---------------------------------------------------------------------------
rem A leftover Streamlit from an earlier run holds the port and makes this
rem launch die with "Port already in use", which reads like a code error.
rem So: walk forward until a free port is found, and say so.
set "ASKED=%PORT%"
set /a ENDPORT=%ASKED%+9
set "FREE="
set "PORTDUMP=%TEMP%\menu_vision_ports.txt"
netstat -ano > "%PORTDUMP%" 2>nul
for /L %%P in (%ASKED%,1,%ENDPORT%) do (
    if not defined FREE (
        findstr /C:":%%P " "%PORTDUMP%" | findstr /C:"LISTENING" >nul
        if errorlevel 1 set "FREE=%%P"
    )
)
del "%PORTDUMP%" >nul 2>&1
if not defined FREE (
    echo [error] No free port between %ASKED% and %ENDPORT%.
    goto :fail
)
set "PORT=%FREE%"
if not "%PORT%"=="%ASKED%" (
    echo [note] Port %ASKED% is taken by another Streamlit - using %PORT% instead.
    echo        Close the old window if you meant to reuse %ASKED%.
    echo.
)

echo [run] http://localhost:%PORT%   ^(Ctrl+C to stop^)
echo.
"%VENV_PY%" -m streamlit run "%PROJECT%streamlit_app.py" --server.port %PORT% --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo [error] Streamlit exited with the error above.
    goto :fail
)
goto :end

rem ---------------------------------------------------------------------------
:fail
echo.
echo Setup or launch failed - read the messages above.
pause
endlocal
exit /b 1

:end
endlocal
exit /b 0
