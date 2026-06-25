#!/bin/bash
set -e

cd /home/runner

if [ ! -f .runner ]; then
    ./config.sh \
        --unattended \
        --url "$RUNNER_URL" \
        --token "$RUNNER_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS"
fi

exec ./run.sh
