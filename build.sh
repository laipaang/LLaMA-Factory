#!/bin/bash

# 当前目录及项目名
TARGET_DIR="$PWD"
PROJECT_NAME=$(basename "$TARGET_DIR")

# Git短hash
SHORT_HASH=$(git rev-parse --short HEAD)

# 日期
DATE=$(date +%Y%m%d)

# 文件名
FILENAME="${PROJECT_NAME}-${SHORT_HASH}-${DATE}.tar.gz"
TEMP_FILE="/tmp/$FILENAME"

# 进入上级目录，打包当前目录（包含目录名）
pushd "$(dirname "$TARGET_DIR")" > /dev/null

tar --exclude-vcs --exclude="$TEMP_FILE" -czf "$TEMP_FILE" "$PROJECT_NAME"

popd > /dev/null

# 移回当前目录
mv "$TEMP_FILE" "$FILENAME"

echo "✅ Created: $FILENAME"
