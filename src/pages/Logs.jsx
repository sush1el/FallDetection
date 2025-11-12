import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Select, Input, Space, Tag, message, Modal, Popconfirm } from 'antd';
import { 
  ReloadOutlined, 
  ClearOutlined, 
  SearchOutlined, 
  DownloadOutlined,
  ExclamationCircleOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import './Logs.css';

const { Option } = Select;
const { confirm } = Modal;

const BACKEND_URL = 'http://localhost:5000';

const Logs = () => {
  const [logs, setLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch incidents from database
  const fetchIncidents = async () => {
    setLoading(true);
    try {
      const statusParam = statusFilter !== 'all' ? `?status=${statusFilter}` : '';
      const response = await fetch(`${BACKEND_URL}/incidents${statusParam}`);
      const data = await response.json();
      
      if (data.success) {
        const formattedLogs = data.incidents.map(incident => ({
          key: incident.id,
          id: incident.id,
          date: incident.date,
          time: incident.time,
          location: incident.location,
          timestamp: incident.timestamp,
          resolved_at: incident.resolved_at,
        }));
        
        setLogs(formattedLogs);
        setFilteredLogs(formattedLogs);
        message.success(`Loaded ${formattedLogs.length} incidents`);
      } else {
        message.error('Failed to fetch incidents');
      }
    } catch (error) {
      console.error('Error fetching incidents:', error);
      message.error('Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchIncidents();
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(() => {
      fetchIncidents();
    }, 5000);
    
    return () => clearInterval(interval);
  }, [statusFilter]);

  // Search filter
  useEffect(() => {
    if (searchTerm) {
      const filtered = logs.filter(log => 
        log.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.date.includes(searchTerm) ||
        log.time.includes(searchTerm)
      );
      setFilteredLogs(filtered);
    } else {
      setFilteredLogs(logs);
    }
  }, [searchTerm, logs]);

  const columns = [
    {
      title: 'Incident ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id) => <Tag color="blue">#{id}</Tag>
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
      sorter: (a, b) => new Date(a.date) - new Date(b.date),
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
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Popconfirm
          title="Delete this incident?"
          description="This action cannot be undone."
          onConfirm={() => handleDelete(record.id)}
          okText="Delete"
          cancelText="Cancel"
          okButtonProps={{ danger: true }}
        >
          <Button 
            size="small" 
            danger
            icon={<DeleteOutlined />}
          >
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ];

  const handleRefresh = () => {
    fetchIncidents();
  };

  const handleDelete = async (incidentId) => {
    try {
      const response = await fetch(`${BACKEND_URL}/incidents/${incidentId}`, {
        method: 'DELETE'
      });
      const data = await response.json();
      
      if (data.success) {
        message.success('Incident deleted successfully');
        fetchIncidents();
      } else {
        message.error('Failed to delete incident');
      }
    } catch (error) {
      console.error('Error deleting incident:', error);
      message.error('Failed to delete incident');
    }
  };

  const handleClearAll = () => {
    confirm({
      title: 'Are you sure you want to clear all incidents?',
      icon: <ExclamationCircleOutlined />,
      content: 'This action cannot be undone. All incident records will be permanently deleted.',
      okText: 'Yes, Clear All',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          const response = await fetch(`${BACKEND_URL}/incidents/clear`, {
            method: 'POST'
          });
          const data = await response.json();
          
          if (data.success) {
            message.success('All incidents cleared');
            setLogs([]);
            setFilteredLogs([]);
          } else {
            message.error('Failed to clear incidents');
          }
        } catch (error) {
          console.error('Error clearing incidents:', error);
          message.error('Failed to clear incidents');
        }
      },
    });
  };

  const handleExport = () => {
    const headers = ['Incident ID', 'Date', 'Time', 'Location'];
    const csvData = [
      headers.join(','),
      ...filteredLogs.map(log => 
        [log.id, log.date, log.time, log.location].join(',')
      )
    ].join('\n');

    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `fall-incidents-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    message.success('Incidents exported successfully!');
  };

  return (
    <div className="logs-container">
      <Card 
        title={<h2>Fall Incident Logs</h2>}
        className="logs-card"
      >
        {/* Filters and Actions */}
        <div className="logs-header">
          <Space wrap style={{ marginBottom: 16 }}>
            <Input
              placeholder="Search by location, date..."
              prefix={<SearchOutlined />}
              style={{ width: 300 }}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            
            <Select 
              value={statusFilter} 
              onChange={setStatusFilter}
              style={{ width: 150 }}
            >
              <Option value="all">All Status</Option>
              <Option value="Active">Active Falls</Option>
              <Option value="Resolved">Resolved Falls</Option>
            </Select>
          </Space>

          <Space>
            <Button 
              icon={<DownloadOutlined />} 
              onClick={handleExport}
              type="primary"
            >
              Export CSV
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={handleRefresh}
              loading={loading}
            >
              Refresh
            </Button>
            <Button 
              icon={<ClearOutlined />} 
              onClick={handleClearAll}
              danger
            >
              Clear All
            </Button>
          </Space>
        </div>

        {/* Logs Table */}
        <Table 
          columns={columns} 
          dataSource={filteredLogs} 
          loading={loading}
          pagination={{ 
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} incidents`
          }}
          scroll={{ x: 800 }}
        />
      </Card>
    </div>
  );
};

export default Logs;