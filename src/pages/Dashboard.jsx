import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Button, Badge, Alert } from 'antd';
import { 
  ReloadOutlined, 
  ClearOutlined, 
  WarningOutlined,
  CameraOutlined,
  WifiOutlined,
  UserOutlined,
  ManOutlined
} from '@ant-design/icons';
import './Dashboard.css';

const BACKEND_URL = 'http://localhost:5000';

const Dashboard = () => {
  // Refs
  const imageRef = React.useRef(null);

  // State management
  const [isLive, setIsLive] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    status: 'Initializing',
    confidence: 0.0,
    people_detected: 0,
    is_fall: false,
    fps: 0
  });
  
  const [incidentLogs, setIncidentLogs] = useState([
    {
      key: '1',
      date: '2025-10-25',
      time: '15:20',
      location: 'Hallway',
      status: 'Fall Detected (Active)',
    },
    {
      key: '2',
      date: '2025-10-26',
      time: '15:20',
      location: 'Room 208',
      status: 'Fall Detected (Active)',
    },
    {
      key: '3',
      date: '2025-10-26',
      time: '15:20',
      location: 'Room 310',
      status: 'Normal',
    },
    {
      key: '4',
      date: '2025-10-26',
      time: '10:90',
      location: 'Room 111',
      status: 'Normal',
    },
  ]);

  const [liveAlerts, setLiveAlerts] = useState([]);

  const [camerasActive, setCamerasActive] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [peopleDetected, setPeopleDetected] = useState(0);
  const [bodyPosture, setBodyPosture] = useState('Initializing');
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Check backend health
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`);
        const data = await response.json();
        setIsConnected(data.status === 'healthy' && data.detector_loaded);
        setIsLive(data.camera_available);
        setCamerasActive(data.camera_available ? 1 : 0);
      } catch (error) {
        console.error('Backend not available:', error);
        setIsConnected(false);
        setIsLive(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch detection status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/status`);
        const data = await response.json();
        setBackendStatus(data);
        
        // Update UI states
        setPeopleDetected(data.people_detected);
        setBodyPosture(data.status);
        setLastUpdate(new Date());
        
        // Add fall alerts
        if (data.is_fall) {
          const newAlert = {
            id: Date.now(),
            message: `FALL DETECTED - Live Camera - ${new Date().toLocaleTimeString()}`,
            time: new Date()
          };
          
          setLiveAlerts(prev => {
            const updated = [newAlert, ...prev];
            return updated.slice(0, 10); // Keep only last 10 alerts
          });
          
          // Add to incident log
          const newLog = {
            key: `log-${Date.now()}`,
            date: new Date().toLocaleDateString(),
            time: new Date().toLocaleTimeString(),
            location: 'Live Camera',
            status: 'Fall Detected (Active)',
          };
          
          setIncidentLogs(prev => {
            // Check if we already have a recent fall log (within last 10 seconds)
            const recentFall = prev.find(log => 
              log.location === 'Live Camera' && 
              log.status.includes('Fall') &&
              Math.abs(new Date() - new Date(`${log.date} ${log.time}`)) < 10000
            );
            
            if (!recentFall) {
              return [newLog, ...prev];
            }
            return prev;
          });
        }
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    const interval = setInterval(fetchStatus, 500); // Update every 500ms
    return () => clearInterval(interval);
  }, []);

  // Update last update time
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const columns = [
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: 'Time',
      dataIndex: 'time',
      key: 'time',
    },
    {
      title: 'Location',
      dataIndex: 'location',
      key: 'location',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <span className={status.includes('Fall') ? 'status-fall' : 'status-normal'}>
          {status}
        </span>
      ),
    },
  ];

  const handleRefresh = () => {
    setLastUpdate(new Date());
    console.log('Refreshing incident logs...');
  };

  const handleClear = () => {
    setIncidentLogs([]);
    setLiveAlerts([]);
  };

  return (
    <div className="dashboard-container">
      <Row gutter={[16, 16]}>
        {/* Left Column - Webcam Feed and Incident Log */}
        <Col xs={24} lg={14}>
          {/* Webcam Feed with AI Detection */}
          <Card className="webcam-card">
            <div className="live-indicator">
              <Badge 
                status={isLive ? "error" : "default"} 
                text={isLive ? "LIVE - AI DETECTION ACTIVE" : "OFFLINE"} 
                className={isLive ? "live-badge" : "offline-badge"}
              />
            </div>
            <div className="webcam-display">
              {isLive ? (
                <img 
                  ref={imageRef}
                  src={`${BACKEND_URL}/video_feed`}
                  alt="AI-Processed Video Feed"
                  className="webcam-video"
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                />
              ) : (
                <div className="webcam-placeholder" style={{ 
                  position: 'absolute', 
                  top: '50%', 
                  left: '50%', 
                  transform: 'translate(-50%, -50%)',
                  textAlign: 'center'
                }}>
                  <CameraOutlined style={{ fontSize: 64, color: '#666' }} />
                  <p style={{ marginTop: 16, color: '#999' }}>
                    {isConnected ? 'Camera initializing...' : 'Backend not connected. Please start the backend server.'}
                  </p>
                </div>
              )}
              <div className="webcam-overlay">
                <div className="detection-label">
                  {backendStatus.status} ({(backendStatus.confidence * 100).toFixed(1)}%)
                </div>
                <div className="camera-id">YOLOv8 + 1D-CNN</div>
              </div>
            </div>
          </Card>

          {/* Incident Log */}
          <Card 
            title="Incident Log" 
            className="incident-log-card"
            extra={
              <div className="log-actions">
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={handleRefresh}
                  className="action-button refresh-button"
                >
                  Refresh
                </Button>
                <Button 
                  icon={<ClearOutlined />} 
                  onClick={handleClear}
                  className="action-button clear-button"
                >
                  Clear
                </Button>
              </div>
            }
          >
            <Table 
              columns={columns} 
              dataSource={incidentLogs} 
              pagination={false}
              size="small"
            />
          </Card>
        </Col>

        {/* Right Column - Status Cards and Alerts */}
        <Col xs={24} lg={10}>
          {/* Live Alerts */}
          <Card title="Live Alerts" className="alerts-card">
            <div className="alerts-container">
              {liveAlerts.length > 0 ? (
                liveAlerts.map((alert) => (
                  <Alert
                    key={alert.id}
                    message={alert.message}
                    type="error"
                    icon={<WarningOutlined />}
                    showIcon
                    className="alert-item"
                  />
                ))
              ) : (
                <Alert
                  message="No Recent Alerts"
                  description="System is monitoring for falls. All clear."
                  type="success"
                  showIcon
                />
              )}
              <div className="last-update">
                Last Update Status: {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
          </Card>

          {/* Status Cards */}
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            {/* Cameras Active */}
            <Col span={24}>
              <Card className="status-card cameras-card">
                <div className="status-content">
                  <CameraOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">Cameras Active</div>
                    <div className="status-value">{camerasActive}</div>
                  </div>
                </div>
              </Card>
            </Col>

            {/* Connection Status */}
            <Col span={24}>
              <Card className={`status-card connection-card ${isConnected ? 'connected' : 'disconnected'}`}>
                <div className="status-content">
                  <WifiOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">Backend Connection</div>
                    <div className="status-value">{isConnected ? 'Connected' : 'Not Connected'}</div>
                  </div>
                </div>
              </Card>
            </Col>

            {/* People Detected */}
            <Col span={24}>
              <Card className="status-card people-card">
                <div className="status-content">
                  <UserOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">People Detected</div>
                    <div className="status-value">{peopleDetected}</div>
                  </div>
                </div>
              </Card>
            </Col>

            {/* Body Posture */}
            <Col span={24}>
              <Card className="status-card posture-card">
                <div className="status-content">
                  <ManOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">Body Posture</div>
                    <div className="status-value">{bodyPosture}</div>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;