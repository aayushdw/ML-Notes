#!/bin/bash

# Sync content from Google Drive and deploy to GitHub
# Usage: ./sync-and-deploy.sh

set -e  # Exit on any error

# Configuration
SOURCE_DIR="/Users/aayushdw/Library/CloudStorage/GoogleDrive-aayushdw@gmail.com/My Drive/Notes/Obsidian/01 - ML & AI Concepts"
CONTENT_DIR="content"
INDEX_FILE="$CONTENT_DIR/index.md"
BACKUP_INDEX="/tmp/index.md.backup"

echo "=== Content Sync & Deploy ==="

# Step 1: Verify source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Make sure Google Drive is synced and accessible."
    exit 1
fi

# Step 2: Backup index.md
echo "Backing up index.md..."
if [ -f "$INDEX_FILE" ]; then
    cp "$INDEX_FILE" "$BACKUP_INDEX"
else
    echo "Warning: No index.md found to backup"
fi

# Step 3: Clear content directory
echo "Clearing content directory..."
rm -rf "$CONTENT_DIR"/*

# Step 4: Copy fresh content from source (excluding CLAUDE.md and GEMINI.md)
echo "Copying content from Google Drive..."
rsync -a --exclude='CLAUDE.md' --exclude='GEMINI.md' "$SOURCE_DIR/" "$CONTENT_DIR/01 - ML & AI Concepts/"

# Step 5: Restore index.md
echo "Restoring index.md..."
if [ -f "$BACKUP_INDEX" ]; then
    cp "$BACKUP_INDEX" "$INDEX_FILE"
    rm "$BACKUP_INDEX"
else
    echo "Warning: No backup index.md to restore"
fi

# Step 6: Build the site
echo "Building site..."
npx quartz build

echo "Build successful!"

# Step 7: Check for changes
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes detected. Skipping commit and push."
    exit 0
fi

# Step 8: Commit and push
echo "Committing changes..."
git add -A
git commit -m "Update content - $(date '+%Y-%m-%d %H:%M:%S')"

echo "Pushing to GitHub..."
git push

echo "=== Done! ==="
