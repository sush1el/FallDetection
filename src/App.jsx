import React, { useState, useEffect } from 'react';
import { Layout, Menu } from 'antd';
import { 
  DashboardOutlined, 
  FileTextOutlined, 
  MonitorOutlined,  
} from '@ant-design/icons';
import Dashboard from './pages/Dashboard';
import Logs from './pages/Logs';
import Monitoring from './pages/Monitoring';
import './App.css';

const { Header, Content } = Layout;

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  // Shared state for active camera feed
  const [activeCameraId, setActiveCameraId] = useState('camera-1');
  const [cameraChangeTimestamp, setCameraChangeTimestamp] = useState(Date.now());

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: 'logs',
      icon: <FileTextOutlined />,
      label: 'Logs',
    },
    {
      key: 'monitoring',
      icon: <MonitorOutlined />,
      label: 'Monitoring',
    },
  ];

  // Function to handle camera switching
  const handleCameraSwitch = (cameraId) => {
    setActiveCameraId(cameraId);
    setCameraChangeTimestamp(Date.now());
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return (
          <Dashboard 
            activeCameraId={activeCameraId}
            cameraChangeTimestamp={cameraChangeTimestamp}
          />
        );
      case 'logs':
        return <Logs />;
      case 'monitoring':
        return (
          <Monitoring 
            activeCameraId={activeCameraId}
            onCameraSwitch={handleCameraSwitch}
          />
        );
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header className="header-container">
        <div className="logo-full-width">
          <img 
            src="/logo.png" 
            alt="CAIretaker - AI-Based Fall Detection System" 
            className="logo-full"
          />
        </div>
      </Header>

      <Layout>
        <Menu
          mode="horizontal"
          selectedKeys={[currentPage]}
          items={menuItems}
          onClick={({ key }) => setCurrentPage(key)}
          className="main-menu"
          style={{
            background: '#2c3e50',
            color: 'white',
            display: 'flex',
            justifyContent: 'flex-start',
            padding: '0 24px',
          }}
        />

        <Content style={{ padding: '24px', background: '#f0f2f5', minHeight: 'calc(100vh - 128px)' }}>
          {renderPage()}
        </Content>
      </Layout>
    </Layout>
  );
}

export default App;