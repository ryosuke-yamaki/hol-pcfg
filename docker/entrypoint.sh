#!/bin/bash
set -e

# Git configuration
git config --global user.name "${GIT_AUTHOR_NAME}"
git config --global user.email "${GIT_AUTHOR_EMAIL}"
git config --global --add safe.directory "${GIT_SAFE_DIRECTORY}"

exec bash
