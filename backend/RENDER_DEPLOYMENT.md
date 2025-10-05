# Deployment Guide for Render.com

This guide provides step-by-step instructions for deploying the Exoplanet ML Model API on Render.com's free tier.

## Prerequisites

- GitHub account
- Render.com account (sign up at https://render.com)
- Code pushed to a GitHub repository

## Quick Start

### Step 1: Push Code to GitHub

If you haven't already, push your code to GitHub:

```bash
cd /path/to/Hunting-Exoplanets
git add .
git commit -m "Prepare backend for deployment"
git push origin main
```

### Step 2: Create Web Service on Render

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Click "New +" → "Web Service"**

3. **Connect Your GitHub Repository**
   - Click "Connect account" if not connected
   - Select your repository: `Hunting-Exoplanets`
   - Click "Connect"

### Step 3: Configure Service Settings

Fill in the following settings:

| Setting | Value |
|---------|-------|
| **Name** | `exoplanet-ml-api` (or your choice) |
| **Region** | Choose closest to your users |
| **Branch** | `main` (or your default branch) |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` |

### Step 4: Add Environment Variables (Optional)

Click "Advanced" and add:

```
PYTHON_VERSION=3.13
```

### Step 5: Add Persistent Disk Storage

⚠️ **CRITICAL**: This step ensures your trained model persists between deploys.

1. Scroll down to **"Disks"** section
2. Click **"Add Disk"**
3. Configure:
   - **Name**: `model-storage`
   - **Mount Path**: `/opt/render/project/src/backend/models`
   - **Size**: `1 GB` (free tier allows up to 1GB)

### Step 6: Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Start your FastAPI server
   - First deployment takes 3-5 minutes

### Step 7: Monitor Deployment

Watch the logs in the Render dashboard:

```
==> Cloning repository...
==> Installing dependencies...
==> Building...
==> Starting service...
🚀 EXOPLANET ML MODEL API - INITIALIZATION
================================================================================
⚠️  No base model found on disk
🔄 Training new base model with optimal hyperparameters...
📥 Fetching training data from NASA Exoplanet Archive...
   ✓ Fetched 9564 rows and 50 columns
   Training samples: 7651
   Positive samples (planets): 2421 (31.6%)
💾 Saving base model to disk...
📊 Base Model Performance:
   Training Set:
      Accuracy:  0.9912 (99.12%)
   Validation Set:
      Accuracy:  0.9893 (98.93%)
✅ Base model trained and saved successfully
🌟 Starting FastAPI server...
```

### Step 8: Test Your Deployment

Your API will be available at: `https://your-service-name.onrender.com`

Test the health endpoint:
```bash
curl https://your-service-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00.000000"
}
```

Test the API documentation at:
- **Swagger UI**: `https://your-service-name.onrender.com/docs`
- **ReDoc**: `https://your-service-name.onrender.com/redoc`

## Important Notes

### Free Tier Limitations

- **Sleep after inactivity**: Service sleeps after 15 minutes of no requests
- **Cold start**: First request after sleep takes ~30 seconds to wake up
- **RAM**: 512 MB (sufficient for this model)
- **Build time**: 750 hours/month of runtime (plenty for most use cases)

### Keep Service Alive (Optional)

To prevent sleeping, you can use a service like [UptimeRobot](https://uptimerobot.com) to ping your `/health` endpoint every 5-10 minutes.

### Model Training on First Deploy

- On first deployment, the server will fetch data from NASA Exoplanet Archive
- Training the base model takes 2-3 minutes
- Subsequent deploys will load the saved model from disk (if persistent storage is configured)

### Updating Your Deployment

When you push changes to GitHub:

1. Render automatically detects the push
2. It rebuilds and redeploys your service
3. Your persistent disk ensures the model isn't lost

To manually redeploy:
- Go to your service dashboard
- Click "Manual Deploy" → "Deploy latest commit"

## Troubleshooting

### Build Fails

**Error**: `Could not find a version that satisfies the requirement...`

**Solution**: Check that `requirements.txt` uses compatible versions:
```txt
fastapi>=0.104.1
uvicorn>=0.24.0
pandas>=2.0.0
python-multipart>=0.0.6
scikit-learn>=1.3.0
numpy>=1.24.0
ydf>=0.5.0
requests>=2.31.0
```

### Service Won't Start

**Error**: `Port already in use`

**Solution**: Ensure your `app.py` reads the port from environment:
```python
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### Model Not Persisting

**Error**: Model retrains on every deploy

**Solution**:
1. Verify disk is mounted at `/opt/render/project/src/backend/models`
2. Check logs to ensure model is being saved
3. Verify `BASE_MODEL_DIR` path in `model_manager.py`

### NASA API Timeout

**Error**: `Failed to fetch training data from API`

**Solution**:
- This is normal on some deployments due to API rate limits
- The server will still start but won't have a base model
- Try manual redeploy or use the `/retrain` endpoint to train a model

## Render.yaml Configuration (Alternative)

You can also use `render.yaml` for infrastructure-as-code deployment:

```yaml
services:
  - type: web
    name: exoplanet-ml-api
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    rootDir: backend
    envVars:
      - key: PYTHON_VERSION
        value: 3.13
    disk:
      name: model-storage
      mountPath: /opt/render/project/src/backend/models
      sizeGB: 1
```

Place this file in your repository root and Render will auto-detect it.

## Next Steps

After deployment:

1. **Update Frontend**: Point your frontend API URL to your Render URL
2. **Test Endpoints**: Try `/inference` and `/retrain` endpoints
3. **Monitor**: Check logs in Render dashboard for errors
4. **Custom Domain** (optional): Add a custom domain in Render settings

## Cost Breakdown

| Feature | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Runtime | 750 hrs/month | Unlimited |
| RAM | 512 MB | 2+ GB |
| Disk | 1 GB | 10+ GB |
| Sleep | After 15 min | No sleep |
| Price | $0/month | $7+/month |

---

For questions or issues, check the [Render documentation](https://render.com/docs) or the main README.md file.
