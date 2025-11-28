#!/bin/bash
# Auto-deployment script for TruthEngine
# Usage: ./deploy.sh "Your commit message"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 TruthEngine Auto-Deployment${NC}"
echo "================================"

# Get commit message from argument or use default
COMMIT_MSG="${1:-Update TruthEngine}"

# Add all changes
echo -e "${GREEN}📦 Adding changes...${NC}"
git add .

# Commit changes
echo -e "${GREEN}💾 Committing: $COMMIT_MSG${NC}"
git commit -m "$COMMIT_MSG"

# Push to GitHub (triggers auto-deployment on Render)
echo -e "${GREEN}☁️  Pushing to GitHub...${NC}"
git push origin main

echo ""
echo -e "${GREEN}✅ Deployed successfully!${NC}"
echo -e "${BLUE}🌐 Your site will update at: https://truthengine-6lq1.onrender.com/${NC}"
echo -e "${BLUE}⏱️  Wait 2-3 minutes for Render to rebuild${NC}"
echo ""
