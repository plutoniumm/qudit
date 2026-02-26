#! /bin/bash

set -euo pipefail;

rootDir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd;
}

hasPython() {
  if command -v python >/dev/null 2>&1; then
    export PATH="$(command -v python3 | xargs dirname):$PATH";
    return 0;
  fi

  echo "Error: python not found in PATH" >&2;
  exit 127;
}

tok() {
  local file="${FILE:-.vscode/token.env}";
  if [[ ! -f "$file" ]]; then
    echo "Error: token file not found: $file" >&2;
    exit 1;
  fi

  cat "$file";
}

build() {
  hasPython;
  local root;
  root="$(rootDir)";
  cd "$root";

  rm -rf build dist qudit.egg-info;
  black ./**/*.py;
  python setup.py bdist_wheel sdist;
  twine check dist/*;
}

deploy() {
  hasPython;
  local root;
  root="$(rootDir)";
  cd "$root";

  local tokval;
  tokval="$(tok)";

  twine upload dist/* -u __token__ -p "$tokval";
  rm -rf build dist qudit.egg-info;
}

test() {
  hasPython;
  local root;
  root="$(rootDir)";
  cd "$root/tests";

  python circuit_V.py;
  python gates.py;
  python gd.py;
  # python ECC.py;
  python algo.py;
  python circuit_M.py;
  python metrics.py;
  python primitives.py;
  python draw.py;
}

prof() {
  hasPython;

  local root;
  root="$(rootDir)";
  cd "$root/tests";

  python -m cProfile -o program.prof bench_fast.py;
  snakeviz program.prof;
}

head() {
  hasPython;
  local root;
  root="$(rootDir)";
  cd "$root";

  python ./view/headers.py;
}

docs() {
  local root;
  root="$(rootDir)";
  cd "$root";

  local subcmd="${1:-}";
  shift || true;

  head;
  case "$subcmd" in
    dev)
      npm run dev "$@"
      ;;
    build)
      npm run build "$@"
      ;;
    *)
      echo "Usage: ./do docs <dev|build> [args...]" >&2
      exit 2
      ;;
  esac
}

help() {
  cat <<'EOF'
Usage: ./do <command> [args...]

Commands:
  build    Build sdist/wheel and run twine checks
  deploy   Upload dist/* to PyPI using token from .vscode/token.env
  test     Run test scripts in ./tests
  prof     Profile bench_fast.py and open snakeviz
  headers  Run python ./view/headers.py
  docs     Run docs via npm (subcommands: dev, build)

Examples:
  ./do build
  ./do test
  ./do head
  ./do docs dev
  ./do docs build
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "$cmd" in
    build|deploy|test|prof|head|docs|help)
      "$cmd" "$@"
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      help
      exit 2
      ;;
  esac
}

main "$@"
