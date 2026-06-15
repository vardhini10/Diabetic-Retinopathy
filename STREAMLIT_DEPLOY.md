# Streamlit Cloud Deployment Guide

## Deploy to Streamlit Cloud in 3 Steps

### Step 1: Go to Streamlit Cloud
👉 **https://share.streamlit.io/**

### Step 2: Sign In with GitHub
- Click "Sign in with GitHub"
- Authorize Streamlit to access your repositories

### Step 3: Deploy Your App
1. Click **"New app"** button (top left)
2. Fill in the deployment details:
   - **Repository**: `vardhini10/Diabetic-Retinopathy`
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
3. Click **"Deploy"**

⏳ **Wait 2-3 minutes** while Streamlit installs dependencies and launches your app.

---

## Your Live App URL

Once deployed, your app will be live at:

🌐 **`https://vardhini10-diabetic-retinopathy.streamlit.app`**

(This URL structure is auto-generated)

---

## After Deployment

✅ **Share your app**: Copy the URL and share with anyone
✅ **Auto-updates**: Every push to GitHub automatically redeploys
✅ **No server management**: Streamlit handles scaling
✅ **Logs**: View deployment logs in your Streamlit Cloud dashboard

---

## Using Your Deployed App

1. **Register**: Create a new account
2. **Login**: Use your credentials
3. **Upload Image**: Upload a retinal fundus image (JPG, PNG, BMP)
4. **Analyze**: Click "Analyze Eye Image"
5. **View Results**: See severity classification with confidence score

---

## Troubleshooting

### Model not loading?
- Ensure `models/cnn_eye_severity_model.h5` is committed to GitHub
- Check file size (should be 100MB+)

### Image upload not working?
- Check image format (JPG, PNG, or BMP)
- Ensure image size < 10MB
- Use actual eye/retinal images

### App crashes?
- Check logs in Streamlit Cloud dashboard
- Verify all dependencies in `requirements.txt`

---

## Need Help?

📚 **Streamlit Docs**: https://docs.streamlit.io/
💬 **Streamlit Community**: https://discuss.streamlit.io/
🐛 **Report Issues**: Create an issue in your GitHub repo

---

**Status**: ✅ Ready for Streamlit Cloud Deployment
