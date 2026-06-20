#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL FALLBACK ONLY. The canonical release path is the tag-triggered CI in
# .github/workflows/publish.yml: push a vX.Y.Z tag that matches pyproject's
# version and it builds + publishes via OIDC (no stored token). Use this script
# only for local/TestPyPI testing; prefer CI to avoid double-publishing.
# ─────────────────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         🚀 Memorizz PyPI Deployment Script                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract version from pyproject.toml
VERSION=$(grep -E '^version = ' pyproject.toml | sed -E 's/version = "([^"]+)"/\1/')
if [ -z "$VERSION" ]; then
    echo "❌ Could not extract version from pyproject.toml"
    exit 1
fi

echo "📌 Detected version: $VERSION"
echo ""

# Check if git is available and we're in a git repo
if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
    GIT_AVAILABLE=true
    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "⚠️  Warning: You have uncommitted changes in your working directory"
        git status --short
        echo ""
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏸️  Stopping. Please commit or stash your changes first."
            exit 1
        fi
    fi
else
    GIT_AVAILABLE=false
    echo "⚠️  Git not available or not in a git repository. Skipping git operations."
    echo ""
fi

# Check if build tools are installed
if ! command -v twine &> /dev/null; then
    echo "❌ twine not found. Installing build tools..."
    pip install --upgrade build twine
fi

# Step 1: Clean previous builds
echo "📦 Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✅ Cleaned!"

# Step 2: Build package
echo ""
echo "🔨 Step 2: Building package..."
python -m build
echo "✅ Built!"

# Step 3: Check package
echo ""
echo "✅ Step 3: Checking package..."
twine check dist/*
echo "✅ Package check passed!"

# Step 4: Show package info
echo ""
echo "📦 Package files created:"
ls -lh dist/

# Step 5: Ask for TestPyPI upload
echo ""
read -p "🧪 Upload to TestPyPI first? (recommended) [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧪 Uploading to TestPyPI..."
    read -sp "TestPyPI API Token (pypi-...): " TEST_TOKEN
    echo ""

    export TWINE_USERNAME=__token__
    export TWINE_PASSWORD=$TEST_TOKEN

    twine upload --repository testpypi dist/*

    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo ""
    echo "📝 Test installation with:"
    echo "   pip install --index-url https://test.pypi.org/simple/ memorizz"
    echo ""
    read -p "✅ TestPyPI upload successful. Ready for production? [y/N]: " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏸️  Stopping. You can upload to production later with:"
        echo "   twine upload dist/*"
        exit 0
    fi
fi

# Step 6: Upload to production PyPI
echo ""
echo "🚀 Uploading to production PyPI..."
read -sp "Production PyPI API Token (pypi-...): " PROD_TOKEN
echo ""

export TWINE_USERNAME=__token__
export TWINE_PASSWORD=$PROD_TOKEN

twine upload dist/*

echo ""
echo "✅ Uploaded to production PyPI!"

# Git operations
if [ "$GIT_AVAILABLE" = true ]; then
    echo ""
    echo "📝 Step 7: Git operations..."

    # Check if version tag already exists
    if git rev-parse "v$VERSION" >/dev/null 2>&1; then
        echo "⚠️  Tag v$VERSION already exists. Skipping tag creation."
    else
        # Stage changes
        echo "📝 Staging changes..."
        git add pyproject.toml
        # Add any other relevant files that might have changed
        git add -u

        # Check if there are changes to commit
        if ! git diff-index --quiet HEAD --; then
            # Commit with version message
            echo "💾 Committing changes..."
            git commit -m "Release version $VERSION

- Update version to $VERSION
- Prepare for PyPI deployment"

            # Create git tag
            echo "🏷️  Creating git tag v$VERSION..."
            git tag -a "v$VERSION" -m "Release version $VERSION"

            # Push commits and tags
            echo "🚀 Pushing to remote repository..."
            read -p "Push commits and tags to remote? [Y/n]: " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                git push
                git push --tags
                echo "✅ Pushed commits and tags to remote!"
            else
                echo "⏸️  Skipped push. You can push later with:"
                echo "   git push && git push --tags"
            fi
        else
            echo "ℹ️  No changes to commit. Creating tag only..."
            # Create git tag even if no commits
            if ! git rev-parse "v$VERSION" >/dev/null 2>&1; then
                git tag -a "v$VERSION" -m "Release version $VERSION"
                read -p "Push tag to remote? [Y/n]: " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    git push --tags
                    echo "✅ Pushed tag to remote!"
                fi
            fi
        fi
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ✅ DEPLOYMENT COMPLETE!                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 Your package is now live on PyPI!"
echo "🔗 https://pypi.org/project/memorizz/"
echo ""
if [ "$GIT_AVAILABLE" = true ]; then
    echo "🏷️  Git tag: v$VERSION"
    echo "📝 Git commit: Release version $VERSION"
    echo ""
fi
echo "📝 Test installation:"
echo "   pip install memorizz --upgrade"
echo ""
