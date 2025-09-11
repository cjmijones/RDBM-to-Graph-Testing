#!/usr/bin/env bash
set -euo pipefail

# --- Verify we're in a git repo ---
git rev-parse --is-inside-work-tree >/dev/null

CURRENT_BRANCH=$(git branch --show-current)

echo "📂 Current branch: ${CURRENT_BRANCH}"
echo "This script will stage ALL changes, prompt you for a properly formatted commit message,"
echo "and then push to origin/${CURRENT_BRANCH}."
echo

# --- Stage everything ---
git add .

# --- Prompt for commit parts ---
read -rp "Enter commit type (build|ci|docs|feat|fix|perf|refactor|style|test|revert): " TYPE
read -rp "Enter scope (e.g., api, chatbot, dashboard) or leave empty: " SCOPE
read -rp "Enter subject (imperative, lowercase, no period): " SUBJECT

echo
echo "Optional: add a detailed body (press ENTER to skip)."
echo "End with an empty line then Ctrl+D when done."
BODY=$(</dev/stdin || true)

# --- Construct commit message ---
if [[ -n "$SCOPE" ]]; then
  HEADER="${TYPE}(${SCOPE}): ${SUBJECT}"
else
  HEADER="${TYPE}: ${SUBJECT}"
fi

# --- Show commit message preview ---
echo
echo "📝 Commit message preview:"
echo "--------------------------------------"
echo "${HEADER}"
if [[ -n "$BODY" ]]; then
  echo
  echo "${BODY}"
fi
echo "--------------------------------------"
echo

read -rp "Does this look correct? [y/N]: " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "❌ Aborted."
  exit 1
fi

# --- Commit ---
if [[ -n "$BODY" ]]; then
  git commit -m "${HEADER}" -m "${BODY}"
else
  git commit -m "${HEADER}"
fi

# --- Push ---
git push -u origin "${CURRENT_BRANCH}"

echo
echo "✅ Changes committed and pushed to origin/${CURRENT_BRANCH}"
