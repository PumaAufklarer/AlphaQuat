#!/bin/bash
set -e

# Defaults
DATA_DIR="${DATA_DIR:-/data}"
SCHEDULE="${SCHEDULE:-0 2 * * *}"
FETCH_SOURCES="${FETCH_SOURCES:-all}"
FEATURE_SET="${FEATURE_SET:-alpha_combined}"
FEATURE_SINCE="${FEATURE_SINCE:-}"
ONE_SHOT="${ONE_SHOT:-false}"

mkdir -p "$DATA_DIR"

if [ -z "$TUSHARE_TOKEN" ]; then
    echo "ERROR: TUSHARE_TOKEN environment variable is required"
    exit 1
fi

# Generate config.yaml
cat > "$DATA_DIR/config.yaml" << EOF
tushare:
  token: "$TUSHARE_TOKEN"
data:
  dir: "$DATA_DIR"
EOF

run_pipeline() {
    echo "[$(date)] Starting data fetch..."
    alpha-quat -c "$DATA_DIR/config.yaml" fetch -s "$FETCH_SOURCES"

    FEATURE_ARGS="-f $FEATURE_SET"
    if [ -n "$FEATURE_SINCE" ]; then
        FEATURE_ARGS="$FEATURE_ARGS --rebuild --since $FEATURE_SINCE"
    fi
    echo "[$(date)] Computing features..."
    alpha-quat -c "$DATA_DIR/config.yaml" feature $FEATURE_ARGS

    echo "[$(date)] Pipeline complete."
}

if [ "$ONE_SHOT" = "true" ]; then
    run_pipeline
    exit 0
fi

# Daemon mode: run immediately on startup, then on schedule
run_pipeline

echo "[$(date)] Scheduling pipeline: $SCHEDULE"
echo "$SCHEDULE root /entrypoint.sh --one-shot >> /var/log/alpha-quat.log 2>&1" > /etc/cron.d/alpha-quat
chmod 0644 /etc/cron.d/alpha-quat
crontab /etc/cron.d/alpha-quat

touch /var/log/alpha-quat.log
cron && tail -f /var/log/alpha-quat.log
