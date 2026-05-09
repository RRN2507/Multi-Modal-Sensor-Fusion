#!/usr/bin/env bash
# ==========================================================
#  setup_repo.sh
#  Run this ONCE to initialize the local + remote GitHub repo.
#  Usage:
#    chmod +x scripts/setup_repo.sh
#    ./scripts/setup_repo.sh <github-username> <repo-name>
# ==========================================================

set -e

GITHUB_USER="${1:-your-username}"
REPO_NAME="${2:-adas-sensor-fusion}"
BRANCH="main"

echo "=== Step 1: Initialize local git repo ==="
git init
git checkout -b $BRANCH

echo "=== Step 2: Add all files ==="
git add .
git commit -m "feat: initial ADAS multi-modal sensor fusion implementation

- Day 1: nuScenes multi-modal data loader (camera, LiDAR, RADAR)
- Day 2: Per-sensor encoders (Swin-T LSS, PointPillars, RADAR PointNet, Thermal CNN)
- Day 3: Weather-adaptive attention gating module
- Day 4: BEV fusion transformer (window attention)
- Day 5: Multi-task heads (CenterPoint det, velocity, segmentation, trajectory)
- Day 6: Training loop (AMP, OneCycleLR, TensorBoard), evaluation, ONNX export, INT8 quantization"

echo "=== Step 3: Create GitHub repo via CLI ==="
echo "  → Make sure you have 'gh' CLI installed: https://cli.github.com/"
echo "  → Run: gh auth login  (if not already authenticated)"

if command -v gh &> /dev/null; then
    gh repo create "$GITHUB_USER/$REPO_NAME" \
        --private \
        --description "Multi-Modal Sensor Fusion for All-Weather ADAS Object Detection" \
        --source=. \
        --remote=origin \
        --push
    echo "=== Repo created and pushed: https://github.com/$GITHUB_USER/$REPO_NAME ==="
else
    echo "=== 'gh' CLI not found. Creating remote manually: ==="
    echo ""
    echo "  1. Go to https://github.com/new"
    echo "  2. Create repo named: $REPO_NAME"
    echo "  3. Then run:"
    echo "       git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "       git push -u origin $BRANCH"
fi

echo ""
echo "=== Git remote branches to set up ==="
echo "  main     — production-ready code"
echo "  develop  — integration branch"
echo "  day/1-data-loader through day/6-training"
echo ""
echo "  Run: git checkout -b develop && git push -u origin develop"
