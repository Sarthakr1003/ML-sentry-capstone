#!/bin/sh
# ML Sentry launcher for Docker users on macOS / Linux.
#   ./mlsentry.sh          first time: setup questions; afterwards: settings menu
#   ./mlsentry.sh setup    the setup questions          ./mlsentry.sh menu   the settings menu
# After the settings are saved, ML Sentry (re)starts in the background.
set -e
cd "$(dirname "$0")"

# Docker would create a folder named .env if the file did not exist yet
[ -f .env ] || touch .env
mkdir -p state

docker compose run --rm mlsentry mlsentry "$@"

docker compose up -d --force-recreate
echo
echo "ML Sentry is watching your model in the background."
echo "  See what it does:   docker compose logs -f"
echo "  Change settings:    ./mlsentry.sh"
