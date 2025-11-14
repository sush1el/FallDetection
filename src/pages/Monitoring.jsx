import React, { useState, useEffect } from 'react';
import { Card, message, Spin } from 'antd';
import { 
  CameraOutlined, 
  TeamOutlined,
  VideoCameraOutlined,
  EnvironmentOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import './Monitoring.css';

const BACKEND_URL = 'http://localhost:5000';

const Monitoring = ({ activeCameraId, onCameraSwitch }) => {
  const [activeCameraIndex, setActiveCameraIndex] = useState(0);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [detectingCameras, setDetectingCameras] = useState(true);
  const [switchingCamera, setSwitchingCamera] = useState(false);
  const [videoKey, setVideoKey] = useState(Date.now());
  const [backendStatus, setBackendStatus] = useState({
    people_detected: 0,
    fps: 0,
    camera_available: false
  });

  // Detect available cameras from backend
  useEffect(() => {
    const detectCameras = async () => {
      setDetectingCameras(true);
      
      try {
        const response = await fetch(`${BACKEND_URL}/available_cameras`);
        const data = await response.json();
        
        if (data.success && data.cameras.length > 0) {
          const cameras = data.cameras.map((index) => ({
            id: `camera-${index}`,
            cameraIndex: index,
            name: `Camera ${index}`,
            location: 'UAC - Exhibit',
            status: 'online',
            isLive: true,
            peopleCount: 0
          }));

          setAvailableCameras(cameras);
          setActiveCameraIndex(data.current_camera || 0);
          message.success(`Detected ${cameras.length} camera(s)`);
        } else {
          message.warning('No cameras detected');
        }
      } catch (error) {
        console.error('Error detecting cameras:', error);
        message.error('Failed to detect cameras from backend');
        
        // Fallback to single camera
        setAvailableCameras([{
          id: 'camera-0',
          cameraIndex: 0,
          name: 'Default Camera',
          location: 'UAC - Exhibit',
          status: 'online',
          isLive: true,
          peopleCount: 0
        }]);
        setActiveCameraIndex(0);
      } finally {
        setDetectingCameras(false);
      }
    };

    detectCameras();
  }, []);

  // Fetch backend status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/status`);
        const data = await response.json();
        setBackendStatus({
          people_detected: data.people_detected || 0,
          fps: data.fps || 0,
          camera_available: true
        });

        // Update active camera people count
        setAvailableCameras(prev => prev.map((cam) => 
          cam.cameraIndex === activeCameraIndex 
            ? { ...cam, peopleCount: data.people_detected || 0 }
            : cam
        ));
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [activeCameraIndex]);

  const handleCameraClick = async (camera) => {
    if (camera.cameraIndex === activeCameraIndex) {
      message.info(`Already viewing ${camera.name}`);
      return;
    }

    setSwitchingCamera(true);
    
    try {
      // Call backend to switch camera
      const response = await fetch(`${BACKEND_URL}/switch_camera`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          camera_index: camera.cameraIndex
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setActiveCameraIndex(camera.cameraIndex);
        setVideoKey(Date.now()); // Force video refresh
        
        if (onCameraSwitch) {
          onCameraSwitch(camera.id);
        }
        
        message.success(`Switched to ${camera.name}`);
      } else {
        message.error(`Failed to switch camera: ${data.message}`);
      }
    } catch (error) {
      console.error('Error switching camera:', error);
      message.error('Failed to switch camera');
    } finally {
      setSwitchingCamera(false);
    }
  };

  const getStatusInfo = (status) => {
    switch(status) {
      case 'online':
        return { dot: 'online', text: 'LIVE', color: '#52c41a' };
      case 'offline':
        return { dot: 'offline', text: 'OFFLINE', color: '#ff4d4f' };
      default:
        return { dot: 'offline', text: 'UNKNOWN', color: '#999' };
    }
  };

  const onlineCameras = availableCameras.filter(c => c.status === 'online').length;
  const totalPeople = backendStatus.people_detected;

  if (detectingCameras) {
    return (
      <div className="monitoring-container">
        <Card 
          title={
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <VideoCameraOutlined style={{ fontSize: 24 }} />
              <span>Multi-Camera Monitoring System</span>
            </div>
          } 
          className="monitoring-card"
        >
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            minHeight: 400,
            gap: 16
          }}>
            <Spin indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />} />
            <h3>Detecting Available Cameras...</h3>
            <p style={{ color: '#666' }}>Checking connected devices...</p>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="monitoring-container">
      <Card 
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <VideoCameraOutlined style={{ fontSize: 24 }} />
            <span>Multi-Camera Monitoring System</span>
          </div>
        } 
        className="monitoring-card"
      >
        {/* Header Info */}
        <div className="monitoring-header">
          <div className="monitoring-info">
            <div className="info-item">
              <CameraOutlined style={{ fontSize: 20, color: '#1890ff' }} />
              <div>
                <div className="info-label">Active Cameras</div>
                <div className="info-value">{onlineCameras} / {availableCameras.length}</div>
              </div>
            </div>
            
            <div className="info-item">
              <TeamOutlined style={{ fontSize: 20, color: '#52c41a' }} />
              <div>
                <div className="info-label">Total People Detected</div>
                <div className="info-value">{totalPeople}</div>
              </div>
            </div>

            <div className="info-item">
              <EnvironmentOutlined style={{ fontSize: 20, color: '#722ed1' }} />
              <div>
                <div className="info-label">Location</div>
                <div className="info-value">UAC - Exhibit</div>
              </div>
            </div>
          </div>
        </div>

        {switchingCamera && (
          <div style={{
            textAlign: 'center',
            padding: 20,
            background: '#e6f7ff',
            borderRadius: 8,
            marginBottom: 16
          }}>
            <Spin /> <span style={{ marginLeft: 12 }}>Switching camera...</span>
          </div>
        )}

        {/* CCTV Grid */}
        {availableCameras.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            padding: 60,
            background: '#f5f5f5',
            borderRadius: 8
          }}>
            <CameraOutlined style={{ fontSize: 64, color: '#ccc', marginBottom: 16 }} />
            <h3>No Cameras Detected</h3>
            <p style={{ color: '#666' }}>Please connect a camera and refresh the page</p>
          </div>
        ) : (
          <div className="cctv-grid">
            {availableCameras.map((camera) => {
              const statusInfo = getStatusInfo(camera.status);
              const isActive = camera.cameraIndex === activeCameraIndex;
              
              return (
                <div 
                  key={camera.id}
                  className={`camera-feed-card ${isActive ? 'active' : ''} ${camera.status === 'offline' ? 'offline' : ''} ${switchingCamera ? 'switching' : ''}`}
                  onClick={() => !switchingCamera && handleCameraClick(camera)}
                >
                  {/* Header */}
                  <div className="camera-feed-header">
                    <div className="camera-info">
                      <div className="camera-id">{camera.name}</div>
                      <div className="camera-location">
                        <EnvironmentOutlined style={{ fontSize: 10, marginRight: 4 }} />
                        {camera.location}
                      </div>
                    </div>
                    <div className="camera-status-badge">
                      <div className={`status-dot ${statusInfo.dot}`}></div>
                      <span className="status-text">{statusInfo.text}</span>
                    </div>
                  </div>

                  {/* Video Feed - Only show for ACTIVE camera */}
                  {isActive && camera.isLive ? (
                    <img 
                      key={videoKey}
                      src={`${BACKEND_URL}/video_feed?t=${videoKey}`}
                      alt={camera.name}
                      className="camera-feed-video"
                      onError={(e) => {
                        console.log(`Error loading feed for ${camera.name}`);
                      }}
                    />
                  ) : (
                    <div className="camera-feed-placeholder">
                      <CameraOutlined style={{ fontSize: 48, color: '#666' }} />
                      <p style={{ color: '#999', marginTop: 12 }}>
                        {switchingCamera ? 'Switching...' : 'Click to switch'}
                      </p>
                    </div>
                  )}

                  {/* Footer */}
                  <div className="camera-feed-footer">
                    <div className="detection-count">
                      <TeamOutlined />
                      <span>
                        {isActive ? `${camera.peopleCount} ${camera.peopleCount === 1 ? 'person' : 'people'}` : '--'}
                      </span>
                    </div>
                    {isActive && (
                      <div className="active-indicator">
                        ACTIVE FEED
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Legend */}
        {availableCameras.length > 0 && (
          <div className="monitoring-legend">
            <div className="legend-item">
              <div className="legend-color active"></div>
              <span className="legend-label">Active Feed (Currently monitoring)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color online"></div>
              <span className="legend-label">Available (Click to switch)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color offline"></div>
              <span className="legend-label">Offline (Connection lost)</span>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Monitoring;