#!/bin/bash
set -euo pipefail
case "${1:-site}" in
  site)
    shift || true
    export JEKYLL_ENV=development
    exec watchmedo auto-restart --debug-force-polling --directory=. --patterns='_config.yaml' --signal SIGTERM -- \
      bundle exec jekyll serve --config _config.yaml,/opt/preview.yaml --host 0.0.0.0 --force_polling --livereload --trace "$@"
    ;;
  build)
    shift
    export JEKYLL_ENV=production
    exec bundle exec jekyll build --trace "$@"
    ;;
  cite)
    shift
    exec python _cite/cite.py "$@"
    ;;
  *) exec "$@" ;;
esac
