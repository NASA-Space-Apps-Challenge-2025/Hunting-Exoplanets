# Deployment Options for YDF Model

Since YDF exceeds Vercel's 250MB serverless function limit, here are your options:

## ✅ RECOMMENDED: Deploy on Railway
1. Go to https://railway.app
2. Click "Start a New Project"
3. Connect your GitHub repo: NASA-Space-Apps-Challenge-2025/Hunting-Exoplanets
4. Railway will auto-detect the Procfile and deploy
5. Get your live URL instantly

**Already configured in your repo:**
- `Procfile` ✓
- `railway.json` ✓
- `requirements.txt` ✓

## Alternative: Render.com (Free)
1. Go to https://render.com
2. New → Web Service
3. Connect GitHub repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

## Keep Vercel Setup
Your Vercel config is saved for future use if needed.
The Docker config can be used for Railway/Render deployment.
