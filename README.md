
# Deploy TryOn Backend to Render

## Quick Setup (5 minutes)

### Step 1: Push to GitHub

```bash
cd backend
git init
git add .
git commit -m "Initial backend deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tryon-backend.git
git push -u origin main
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Sign up with GitHub (recommended)

### Step 3: Deploy

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repo
3. Configure:
   - **Name:** `tryon-backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free`

### Step 4: Set Environment Variables

Go to **Environment** tab and add:

| Key                     | Value           |
| ----------------------- | --------------- |
| `FAL_API_KEY`           | Your fal.ai key |
| `CLOUDINARY_CLOUD_NAME` | `dea9nvcwq`     |
| `CLOUDINARY_API_KEY`    | Your API key    |
| `CLOUDINARY_API_SECRET` | Your API secret |

### Step 5: Deploy

Click **"Create Web Service"** → Wait 5-10 minutes for first build

---

## Your URL

After deployment:

```
https://tryon-backend.onrender.com
```

## Test It

```
https://tryon-backend.onrender.com/docs
```

## Update Flutter

Change API base URL in Flutter from:

```dart
const baseUrl = 'http://localhost:8000';
```

To:

```dart
const baseUrl = 'https://tryon-backend.onrender.com';
```
