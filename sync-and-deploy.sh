#!/bin/bash

# Sync content from Google Drive and deploy to GitHub
# Usage: ./sync-and-deploy.sh

set -e  # Exit on any error

# Configuration
SOURCE_DIR="/Users/aayushdw/Library/CloudStorage/GoogleDrive-aayushdw@gmail.com/My Drive/Notes/Obsidian/01 - ML & AI Concepts"
CONTENT_DIR="content"

echo "=== Content Sync & Deploy ==="

# Step 1: Verify source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Make sure Google Drive is synced and accessible."
    exit 1
fi

# Step 2: Sync content (only updates changed files, preserves timestamps, removes deleted files)
echo "Syncing content from Google Drive..."
rsync -a --delete \
    --exclude='CLAUDE.md' \
    --exclude='GEMINI.md' \
    --exclude='AGENT.md' \
    "$SOURCE_DIR/" "$CONTENT_DIR/01 - ML & AI Concepts/"

# Step 3: Build the site
echo "Building site..."
npx quartz build

echo "Build successful!"

# Step 4: Check for changes
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes detected. Skipping commit and push."
    exit 0
fi

# Step 5: Commit and push
echo "Committing changes..."
git add -A
git commit -m "Update content - $(date '+%Y-%m-%d %H:%M:%S')"

echo "Pushing to GitHub..."
git push

echo "=== Done! ==="
