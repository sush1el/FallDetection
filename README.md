# CAIretaker Dashboard

AI-Based Fall Detection System Dashboard built with React, Vite, and Ant Design.

![CAIretaker](https://via.placeholder.com/800x400?text=CAIretaker+Dashboard)

## Features

### ✨ Core Features
- **Real-time Dashboard** with webcam feed and live status monitoring
- **Incident Logging System** with advanced filtering and export capabilities
- **Live Alerts** for immediate fall detection notifications
- **System Status Monitoring** (cameras, connection, people detected, body posture)
- **Multi-page Navigation** (Dashboard, Logs, Monitoring, Settings, Live Camera)
- **Responsive Design** optimized for desktop, tablet, and mobile devices

### 📊 Dashboard Components
- Live webcam feed with pose detection overlay
- Real-time incident log table
- Live alerts panel with timestamp
- Status cards:
  - Cameras Active counter
  - Connection Status detector
  - People Detected counter
  - Body Posture indicator

### 📝 Logs Page
- Comprehensive incident history
- Advanced filtering (date, location, status, severity)
- Search functionality
- Export to CSV/PDF capability
- Sortable columns

## Tech Stack

- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **Ant Design 5** - Enterprise-grade UI components
- **MediaDevices API** - Webcam access
- **CSS3** - Custom styling

## Prerequisites

- Node.js 16.x or higher
- npm or yarn package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Quick Start

### 1. Clone/Download the Project
```bash
# If you have this as a zip file, extract it first
cd cairetaker-dashboard
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Add Your Logo (Optional)
Place your logo file in the `public/` folder:
```
public/logo.png
```
Recommended size: 48x48px or larger

### 4. Start Development Server
```bash
npm run dev
```

The app will automatically open at http://localhost:3000

### 5. Build for Production
```bash
npm run build
```
Production files will be in the `dist/` folder

## Project Structure

```
cairetaker-dashboard/
├── public/
│   └── logo.png              # Your logo file
├── src/
│   ├── components/           # Reusable components
│   │   ├── WebcamFeed.jsx
│   │   └── WebcamFeed.css
│   ├── pages/                # Page components
│   │   ├── Dashboard.jsx
│   │   ├── Dashboard.css
│   │   ├── Logs.jsx
│   │   ├── Logs.css
│   │   ├── Monitoring.jsx
│   │   ├── Monitoring.css
│   │   ├── Settings.jsx
│   │   └── Settings.css
│   ├── App.jsx               # Main app component
│   ├── App.css              # App styles
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── README.md               # This file
```

## Available Scripts

```bash
npm run dev      # Start development server (port 3000)
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Check code quality
```

## Usage Guide

### Navigation
- **Dashboard** - Main view with live feed and status
- **Logs** - View and filter incident history
- **Monitoring** - Multi-camera view (coming soon)
- **Settings** - System configuration (coming soon)
- **Live Camera** - Full-screen webcam feed

### Dashboard Features

#### Live Indicator
- 🔴 **Red "LIVE"** - Webcam is actively streaming
- ⚪ **Gray "OFFLINE"** - No webcam detected

#### Incident Log
- View recent incidents with date, time, location, status
- **Refresh** - Update the log with latest data
- **Clear** - Remove all entries

#### Live Alerts
- Real-time notifications when falls are detected
- Timestamp of last update shown at bottom

#### Status Cards
- **Cameras Active** - Number of connected cameras (green)
- **Connection Status** - Internet connectivity (blue/red)
- **People Detected** - Current person count (purple)
- **Body Posture** - Current position (orange)

### Logs Page Features
- Search through incident history
- Filter by date range, location, status
- Sort by any column
- Export logs to file

## Customization

### Colors
Edit CSS files to change the color scheme:
- Primary: `#1890ff`
- Success: `#52c41a`
- Error: `#ff4d4f`
- Dark: `#1a2332`

### Adding Real Data
Currently using mock data. To connect real backend:

1. **Create API service** (`src/services/api.js`):
```javascript
export const fetchIncidents = async () => {
  const response = await fetch('/api/incidents');
  return response.json();
};

export const fetchSystemStatus = async () => {
  const response = await fetch('/api/status');
  return response.json();
};
```

2. **Use in components**:
```javascript
import { fetchIncidents } from '../services/api';

useEffect(() => {
  fetchIncidents().then(data => setIncidentLogs(data));
}, []);
```

3. **WebSocket for live updates**:
```javascript
const ws = new WebSocket('ws://your-server/alerts');
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  setLiveAlerts(prev => [...prev, alert]);
};
```

## Camera Permissions

For webcam to work:
- Use **HTTPS** in production (required by browsers)
- Grant camera permissions when prompted
- Check browser console if issues occur

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Troubleshooting

### Port Already in Use
Change port in `vite.config.js`:
```javascript
server: {
  port: 3001  // Use different port
}
```

### Camera Not Working
1. Check browser permissions
2. Ensure HTTPS (required in production)
3. Try different browser
4. Check browser console for errors

### Build Errors
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Next Steps

### For Full Implementation:
1. Set up MediaPipe Pose Detection backend
2. Create REST API endpoints
3. Implement WebSocket for real-time updates
4. Add authentication system
5. Connect to database
6. Deploy to production server

### Backend API Endpoints Needed:
- `GET /api/incidents` - Fetch incident logs
- `POST /api/incidents/clear` - Clear logs
- `GET /api/status` - Get system status
- `GET /api/cameras` - Get camera list
- `WebSocket /ws/alerts` - Real-time alerts

## Contributing

This is a proprietary project for CAIretaker Fall Detection System.

## License

Copyright © 2025 CAIretaker. All rights reserved.

## Support

For issues or questions, please contact the development team.

---

**Made with ❤️ for CAIretaker**
