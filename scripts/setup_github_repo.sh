#!/usr/bin/env bash
set -euo pipefail

# Быстрая публикация репозитория slevak228/gpttest на GitHub
git init -b main
git add .
git commit -m "feat: modular calls-only microgpt pipeline + GitHub Actions CI"

if command -v gh >/dev/null 2>&1; then
  gh repo create "slevak228/gpttest" --public --source=. --remote=origin --push
else
  git remote add origin "https://github.com/slevak228/gpttest.git"
  git push -u origin main
fi
