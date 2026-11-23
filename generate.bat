@echo off
setlocal
py "%~dp0generate.py" %*
exit /b %errorlevel%
