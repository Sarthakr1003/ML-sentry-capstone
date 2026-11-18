@echo off
rem ML Sentry launcher for Docker users on Windows.
rem   .\mlsentry.cmd         first time: setup questions; afterwards: settings menu
rem   .\mlsentry.cmd setup   the setup questions          .\mlsentry.cmd menu   the settings menu
rem After the settings are saved, ML Sentry (re)starts in the background.
cd /d "%~dp0"

rem Docker would create a folder named .env if the file did not exist yet
if not exist .env type nul > .env
if not exist state mkdir state

docker compose run --rm mlsentry mlsentry %*
if %errorlevel% neq 0 exit /b %errorlevel%

docker compose up -d --force-recreate
echo.
echo ML Sentry is watching your model in the background.
echo   See what it does:   docker compose logs -f
echo   Change settings:    .\mlsentry.cmd
