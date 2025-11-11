import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Button, Badge, Alert, Tag } from 'antd';
import { 
  ReloadOutlined, 
  ClearOutlined, 
  WarningOutlined,
  CameraOutlined,
  WifiOutlined,
  UserOutlined,
  TeamOutlined
} from '@ant-design/icons';
import './Dashboard.css';

const BACKEND_URL = 'http://localhost:5000';

const Dashboard = () => {
  const imageRef = React.useRef(null);

  const [isLive, setIsLive] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    people_detected: 0,
    detections: [],
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
  const [detectedPeople, setDetectedPeople] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Force video feed request on mount to ensure camera starts
  useEffect(() => {
    if (isLive && imageRef.current) {
      // Force reload the image source to trigger video feed
      imageRef.current.src = `${BACKEND_URL}/video_feed?t=${Date.now()}`;
    }
  }, [isLive]);

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
        
        setPeopleDetected(data.people_detected);
        setDetectedPeople(data.detections || []);
        setLastUpdate(new Date());
        
        // Check for falls in any detected person
        const fallDetected = data.detections?.some(d => d.is_fall) || false;
        
        if (fallDetected) {
          const fallPerson = data.detections.find(d => d.is_fall);
          const newAlert = {
            id: Date.now(),
            message: `FALL DETECTED - Person ID: ${fallPerson.id} - ${new Date().toLocaleTimeString()}`,
            time: new Date()
          };
          
          setLiveAlerts(prev => {
            const updated = [newAlert, ...prev];
            return updated.slice(0, 10);
          });
          
          const newLog = {
            key: `log-${Date.now()}`,
            date: new Date().toLocaleDateString(),
            time: new Date().toLocaleTimeString(),
            location: `Live Camera - Person ID: ${fallPerson.id}`,
            status: 'Fall Detected (Active)',
          };
          
          setIncidentLogs(prev => {
            const recentFall = prev.find(log => 
              log.location.includes(`Person ID: ${fallPerson.id}`) && 
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

    const interval = setInterval(fetchStatus, 500);
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

  const getStatusColor = (status) => {
    if (status === 'Fallen') return 'red';
    if (status === 'Sitting') return 'orange';
    if (status === 'Standing') return 'green';
    return 'default';
  };

  return (
    <div className="dashboard-container">
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          {/* Webcam Feed with AI Detection */}
          <Card className="webcam-card">
            <div className="live-indicator">
              <Badge 
                status={isLive ? "error" : "default"} 
                text={isLive ? "LIVE - MULTI-PERSON TRACKING ACTIVE" : "OFFLINE"} 
                className={isLive ? "live-badge" : "offline-badge"}
              />
            </div>
            <div className="webcam-display">
              {isLive ? (
                <img 
                  ref={imageRef}
                  src={`${BACKEND_URL}/video_feed?t=${Date.now()}`}
                  alt="AI-Processed Video Feed"
                  className="webcam-video"
                  onError={(e) => {
                    console.log('Video feed error, retrying in 2 seconds...');
                    setTimeout(() => {
                      e.target.src = `${BACKEND_URL}/video_feed?t=${Date.now()}`;
                    }, 2000);
                  }}
                  onLoad={() => console.log('Video feed loaded successfully')}
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
            </div>
          </Card>

          {/* Detected People Table */}
          {detectedPeople.length > 0 && (
            <Card 
              title="Currently Detected People" 
              style={{ marginTop: 16 }}
              className="incident-log-card"
            >
              <Table 
                dataSource={detectedPeople.map(p => ({ ...p, key: p.id }))}
                pagination={false}
                size="small"
                columns={[
                  {
                    title: 'Person ID',
                    dataIndex: 'id',
                    key: 'id',
                    render: (id) => <Tag color="blue">ID: {id}</Tag>
                  },
                  {
                    title: 'Status',
                    dataIndex: 'status',
                    key: 'status',
                    render: (status, record) => (
                      <Tag color={getStatusColor(status)}>
                        {status} {record.confidence > 0 && `(${(record.confidence * 100).toFixed(0)}%)`}
                      </Tag>
                    )
                  },
                  {
                    title: 'Alert',
                    dataIndex: 'is_fall',
                    key: 'is_fall',
                    render: (isFall) => 
                      isFall ? (
                        <Tag color="red" icon={<WarningOutlined />}>FALL DETECTED</Tag>
                      ) : (
                        <Tag color="success">Normal</Tag>
                      )
                  }
                ]}
              />
            </Card>
          )}

          {/* Incident Log */}
          <Card 
            title="Incident Log" 
            className="incident-log-card"
            style={{ marginTop: 16 }}
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

            <Col span={24}>
              <Card className="status-card people-card">
                <div className="status-content">
                  <TeamOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">People Detected</div>
                    <div className="status-value">{peopleDetected}</div>
                  </div>
                </div>
              </Card>
            </Col>

            <Col span={24}>
              <Card className="status-card posture-card">
                <div className="status-content">
                  <UserOutlined className="status-icon" />
                  <div className="status-info">
                    <div className="status-label">FPS</div>
                    <div className="status-value">{backendStatus.fps || 0}</div>
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