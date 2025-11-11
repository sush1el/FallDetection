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

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'logs':
        return <Logs />;
      case 'monitoring':
        return <Monitoring />;
      case 'settings':
        return <Settings />;
      case 'livecamera':
        return <WebcamFeed />;
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
