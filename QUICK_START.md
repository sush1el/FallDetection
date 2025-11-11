# 🚀 Quick Start Guide - CAIretaker Dashboard

## Installation in 3 Steps

### Step 1: Install Node.js
If you don't have Node.js installed:
1. Go to https://nodejs.org/
2. Download and install the LTS version (18.x or 20.x)
3. Verify installation:
```bash
node --version
npm --version
```

### Step 2: Install Dependencies
Open terminal in the project folder and run:
```bash
npm install
```

This will take 1-2 minutes and install all required packages.

### Step 3: Start the Application
```bash
npm run dev
```

The browser will automatically open at http://localhost:3000

**That's it! You're running! 🎉**

---

## Optional: Add Your Logo

1. Place your logo file in the `public/` folder
2. Name it `logo.png`
3. Recommended size: 48x48px or larger
4. The logo will appear in the top-left corner

---

## Troubleshooting

### "Port 3000 is already in use"
Open `vite.config.js` and change the port:
```javascript
server: {
  port: 3001  // Change to any available port
}
```

### "Cannot find module"
Delete and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Camera Not Working
- Click "Allow" when browser asks for camera permission
- Make sure no other application is using the camera
- Try reloading the page

---

## Understanding the Dashboard

### Navigation Bar
- **Dashboard** → Main view with live feed and alerts
- **Logs** → View past incident records
- **Monitoring** → Coming soon
- **Settings** → Coming soon
- **Live Camera** → Full-screen webcam view

### Live Indicator
- 🔴 **LIVE** (Red) = Camera is active
- ⚪ **OFFLINE** (Gray) = Camera is off

### Status Cards (Right Side)
- **Cameras Active** - How many cameras are running
- **Connection Status** - Internet connection state
- **People Detected** - Number of people in view
- **Body Posture** - Current position (Standing/Sitting/Lying/Fallen)

---

## Next Steps for Production

Currently, the app uses **mock data** (fake data for testing). To make it work with real cameras and detection:

### 1. Backend Setup Required:
- MediaPipe Pose Detection server
- REST API for data
- WebSocket for real-time updates
- Database for storing logs

### 2. Connect to Backend:
Replace mock data in components with real API calls.

### 3. Deploy:
```bash
npm run build
```
Upload the `dist/` folder to your web server.

---

## Common Commands

```bash
npm run dev      # Start development server
npm run build    # Create production build
npm run preview  # Preview production build
```

---

## Need Help?

1. Check the `README.md` for detailed documentation
2. Look at browser console (F12) for error messages
3. Ensure all files are in correct folders
4. Make sure Node.js version is 16 or higher

---

**Happy coding! 🎯**
