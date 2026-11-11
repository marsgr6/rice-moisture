#!/bin/bash
while IFS=$'\t' read -r src dst; do [ -f "$dst" ] || convert "$src" -resize 15% -depth 8 "$dst"; done
