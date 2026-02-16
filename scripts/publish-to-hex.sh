#!/usr/bin/env bash
set -euo pipefail

# Publish faber_nn_nifs to hex.pm
# Usage: ./scripts/publish-to-hex.sh
#
# Requires: ~/.config/rebar3/hex.config with api_key set

cd "$(dirname "$0")/.."

VERSION=$(grep -oP '{vsn,\s*"\K[^"]+' src/faber_nn_nifs.app.src)
echo "==> Publishing faber_nn_nifs v${VERSION} to hex.pm..."

echo "==> Building faber_nn_nifs..."
rebar3 compile

echo "==> Running tests..."
rebar3 eunit

echo "==> Publishing to hex.pm..."
rebar3 hex publish --yes

echo "==> Done! faber_nn_nifs v${VERSION} published to hex.pm"
echo "==> View at: https://hex.pm/packages/faber_nn_nifs"
