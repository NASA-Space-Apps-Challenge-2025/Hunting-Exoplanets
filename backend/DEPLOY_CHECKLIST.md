# Render.com Deployment Checklist ✅

Use this checklist to ensure your deployment is configured correctly.

## Pre-Deployment

- [ ] Push all code to GitHub
- [ ] Verify `requirements.txt` is up to date
- [ ] Verify `app.py` uses `PORT` environment variable
- [ ] Test API locally one more time

## Render Configuration

- [ ] Create new Web Service on Render
- [ ] Connect GitHub repository
- [ ] Set **Root Directory** to `backend`
- [ ] Set **Build Command** to `pip install -r requirements.txt`
- [ ] Set **Start Command** to `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] Select **Free** instance type

## Persistent Storage (CRITICAL!)

- [ ] Add disk storage in "Disks" section
- [ ] Name: `model-storage`
- [ ] Mount Path: `/opt/render/project/src/backend/models`
- [ ] Size: `1 GB`

## Environment Variables (Optional)

- [ ] Set `PYTHON_VERSION=3.13`

## Post-Deployment

- [ ] Monitor deployment logs
- [ ] Wait for base model to train (2-3 minutes on first deploy)
- [ ] Test `/health` endpoint: `https://your-service.onrender.com/health`
- [ ] Test API docs: `https://your-service.onrender.com/docs`
- [ ] Test `/info` endpoint to verify model is loaded
- [ ] Update frontend with production API URL

## Optional Enhancements

- [ ] Set up UptimeRobot to prevent service from sleeping
- [ ] Add custom domain in Render settings
- [ ] Configure CORS origins for production (update `app.py`)

## Files Created for Deployment

✅ **render.yaml** - Infrastructure-as-code configuration (optional)
✅ **runtime.txt** - Specifies Python version
✅ **RENDER_DEPLOYMENT.md** - Detailed deployment guide
✅ **requirements.txt** - Python dependencies
✅ **app.py** - Configured with PORT environment variable

## Quick Commands

### Test Locally
```bash
cd backend
python app.py
```

### Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Test Production Health
```bash
curl https://your-service-name.onrender.com/health
```

### Test Production Inference
```bash
curl -X POST "https://your-service-name.onrender.com/inference" \
  -F "file=@test_data.csv"
```

## Troubleshooting

### Service won't start
- Check logs in Render dashboard
- Verify build command completed successfully
- Ensure all dependencies in `requirements.txt`

### Model retrains every deploy
- Verify disk is mounted correctly
- Check mount path: `/opt/render/project/src/backend/models`
- Check logs for "Base model loaded from disk" message

### NASA API timeout
- Normal on first deploy due to rate limits
- Manually redeploy or use `/retrain` endpoint

---

For detailed instructions, see [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)
