import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Button, Badge, Alert, Tag } from 'antd';
import { 
  ReloadOutlined, 
  ClearOutlined, 
  WarningOutlined,
  CameraOutlined,
  WifiOutlined,
  UserOutlined,
  TeamOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import './Dashboard.css';

const BACKEND_URL = 'http://localhost:5000';

const Dashboard = () => {
  const imageRef = React.useRef(null);
  const [videoFeedLoaded, setVideoFeedLoaded] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    people_detected: 0,
    detections: [],
    fps: 0
  });

  const [liveAlerts, setLiveAlerts] = useState([]);
  const [camerasActive, setCamerasActive] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [peopleDetected, setPeopleDetected] = useState(0);
  const [detectedPeople, setDetectedPeople] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [recentIncidents, setRecentIncidents] = useState([]);

  // Track which person IDs we've already alerted for
  const alertedFallsRef = React.useRef(new Set());

  // Fetch recent incidents
  const fetchRecentIncidents = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/incidents?limit=10`);
      const data = await response.json();
      
      if (data.success) {
        const formatted = data.incidents.map(incident => ({
          key: incident.id,
          id: incident.id,
          date: incident.date,
          time: incident.time,
          location: incident.location,
        }));
        setRecentIncidents(formatted);
      }
    } catch (error) {
      console.error('Error fetching recent incidents:', error);
    }
  };

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

  // Fetch recent incidents
  useEffect(() => {
    fetchRecentIncidents();
    const interval = setInterval(fetchRecentIncidents, 5000);
    return () => clearInterval(interval);
  }, []);

  // Trigger video feed load when backend is ready
  useEffect(() => {
    if (isConnected && !videoFeedLoaded && imageRef.current) {
      console.log('Backend connected, starting video feed...');
      imageRef.current.src = `${BACKEND_URL}/video_feed?t=${Date.now()}`;
      setVideoFeedLoaded(true);
    }
  }, [isConnected, videoFeedLoaded]);

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
        
        // Check for NEW falls (not already alerted)
        if (data.detections) {
          data.detections.forEach(person => {
            if (person.is_fall && person.incident_id) {
              // Only alert if we haven't alerted for this incident ID before
              if (!alertedFallsRef.current.has(person.incident_id)) {
                const newAlert = {
                  id: person.incident_id,
                  person_id: person.id,
                  message: `FALL DETECTED - Person ID: ${person.id} - Incident #${person.incident_id}`,
                  time: new Date()
                };
                
                setLiveAlerts(prev => {
                  const updated = [newAlert, ...prev];
                  return updated.slice(0, 10);
                });
                
                // Mark this incident as alerted
                alertedFallsRef.current.add(person.incident_id);
                
                console.log(`🚨 NEW FALL ALERT: Person ${person.id}, Incident #${person.incident_id}`);
              }
            }
          });
        }
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    const interval = setInterval(fetchStatus, 500);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setLastUpdate(new Date());
    // Force refresh video feed
    if (imageRef.current && isConnected) {
      setVideoFeedLoaded(false);
      setTimeout(() => {
        if (imageRef.current) {
          imageRef.current.src = `${BACKEND_URL}/video_feed?t=${Date.now()}`;
          setVideoFeedLoaded(true);
        }
      }, 100);
    }
    console.log('Refreshing dashboard...');
  };

  const handleClearAlerts = () => {
    setLiveAlerts([]);
    alertedFallsRef.current.clear();
  };

  const getStatusColor = (status) => {
    if (status === 'Fallen') return 'red';
    if (status === 'Sitting') return 'orange';
    if (status === 'Standing') return 'green';
    return 'default';
  };

  const handleResolveIncident = async (incidentId) => {
    try {
      const response = await fetch(`${BACKEND_URL}/incidents/${incidentId}/resolve`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        console.log(`Incident ${incidentId} resolved`);
        // Remove from alerts
        setLiveAlerts(prev => prev.filter(alert => alert.id !== incidentId));
      }
    } catch (error) {
      console.error('Error resolving incident:', error);
    }
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
              {isConnected ? (
                <img 
                  ref={imageRef}
                  alt="AI-Processed Video Feed"
                  className="webcam-video"
                  onError={(e) => {
                    console.log('Video feed error, retrying in 3 seconds...');
                    setTimeout(() => {
                      if (e.target) {
                        e.target.src = `${BACKEND_URL}/video_feed?t=${Date.now()}`;
                      }
                    }, 3000);
                  }}
                  onLoad={() => {
                    console.log('Video feed loaded successfully');
                    setIsLive(true);
                  }}
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

          {/* Currently Detected People */}
          <Card 
            title="Live Monitoring" 
            style={{ marginTop: 16 }}
            className="incident-log-card"
            extra={<Badge count={detectedPeople.length} showZero />}
          >
            {detectedPeople.length > 0 ? (
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
                    render: (isFall, record) => 
                      isFall ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <Tag color="red" icon={<WarningOutlined />}>
                            FALLEN - Incident #{record.incident_id}
                          </Tag>
                          <Button 
                            size="small" 
                            type="primary"
                            icon={<CheckCircleOutlined />}
                            onClick={() => handleResolveIncident(record.incident_id)}
                          >
                            Resolve
                          </Button>
                        </div>
                      ) : (
                        <Tag color="success">Normal</Tag>
                      )
                  }
                ]}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: 20, color: '#999' }}>
                No people detected in frame
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          {/* Live Alerts */}
          <Card 
            title="Live Fall Alerts" 
            className="alerts-card"
            extra={
              <Button 
                size="small" 
                icon={<ClearOutlined />} 
                onClick={handleClearAlerts}
              >
                Clear
              </Button>
            }
          >
            <div className="alerts-container">
              {liveAlerts.length > 0 ? (
                liveAlerts.map((alert) => (
                  <Alert
                    key={alert.id}
                    message={
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>{alert.message}</span>
                        <Button 
                          size="small" 
                          type="link"
                          onClick={() => handleResolveIncident(alert.id)}
                        >
                          Dismiss
                        </Button>
                      </div>
                    }
                    description={alert.time.toLocaleString()}
                    type="error"
                    icon={<WarningOutlined />}
                    showIcon
                    className="alert-item"
                  />
                ))
              ) : (
                <Alert
                  message="No Active Fall Alerts"
                  description="System is monitoring for falls. All clear."
                  type="success"
                  showIcon
                />
              )}
              
            </div>
          </Card>

          {/* Recent Logs */}
          <Card 
            title= "Recent Logs" 
            style={{ marginTop: 16 }}
            extra={
              <Button 
                size="small" 
                icon={<ReloadOutlined />} 
                onClick={fetchRecentIncidents}
              >
                Refresh
              </Button>
            }
          >
            
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              {recentIncidents.length > 0 ? (
                <Table 
                  dataSource={recentIncidents}
                  pagination={false}
                  size="small"
                  showHeader={true}
                  columns={[
                    {
                      title: 'ID',
                      dataIndex: 'id',
                      key: 'id',
                      width: 60,
                      render: (id) => <Tag color="blue">#{id}</Tag>
                    },
                    {
                      title: 'Date',
                      dataIndex: 'date',
                      key: 'date',
                      width: 100,
                    },
                    {
                      title: 'Time',
                      dataIndex: 'time',
                      key: 'time',
                      width: 80,
                    },
                    {
                      title: 'Location',
                      dataIndex: 'location',
                      key: 'location',
                      ellipsis: true,
                    }
                  ]}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: 20, color: '#999' }}>
                  No incidents recorded yet
                </div>,
                <div className="last-update">
                Last Update: {lastUpdate.toLocaleTimeString()}
                </div>
                
              )}
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