@echo off
rem Set up the S-Bahn simulation on this Windows machine. Needs Python 3.8+ (python.org) on PATH.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 setup.py %*) else (python setup.py %*)
