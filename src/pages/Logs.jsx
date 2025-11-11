import React, { useState } from 'react';
import { Card, Table, Button, DatePicker, Select, Input, Space } from 'antd';
import { ReloadOutlined, ClearOutlined, SearchOutlined, DownloadOutlined } from '@ant-design/icons';
import './Logs.css';

const { RangePicker } = DatePicker;
const { Option } = Select;

const Logs = () => {
  const [logs, setLogs] = useState([
    {
      key: '1',
      date: '2025-10-25',
      time: '15:20',
      location: 'Hallway',
      status: 'Fall Detected (Active)',
      severity: 'Critical',
      response: 'Alert Sent',
    },
    {
      key: '2',
      date: '2025-10-26',
      time: '15:20',
      location: 'Room 208',
      status: 'Fall Detected (Active)',
      severity: 'Critical',
      response: 'Alert Sent',
    },
    {
      key: '3',
      date: '2025-10-26',
      time: '15:20',
      location: 'Room 310',
      status: 'Normal',
      severity: 'Low',
      response: 'None',
    },
    {
      key: '4',
      date: '2025-10-26',
      time: '10:90',
      location: 'Room 111',
      status: 'Normal',
      severity: 'Low',
      response: 'None',
    },
    {
      key: '5',
      date: '2025-10-24',
      time: '09:15',
      location: 'Kitchen',
      status: 'Fall Detected (Resolved)',
      severity: 'Medium',
      response: 'Staff Responded',
    },
  ]);

  const [filteredLogs, setFilteredLogs] = useState(logs);

  const columns = [
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
      filters: [
        { text: 'Hallway', value: 'Hallway' },
        { text: 'Room 208', value: 'Room 208' },
        { text: 'Room 310', value: 'Room 310' },
        { text: 'Room 111', value: 'Room 111' },
        { text: 'Kitchen', value: 'Kitchen' },
      ],
      onFilter: (value, record) => record.location.includes(value),
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
      filters: [
        { text: 'Fall Detected', value: 'Fall' },
        { text: 'Normal', value: 'Normal' },
      ],
      onFilter: (value, record) => record.status.includes(value),
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <span className={`severity-${severity.toLowerCase()}`}>
          {severity}
        </span>
      ),
    },
    {
      title: 'Response',
      dataIndex: 'response',
      key: 'response',
    },
  ];

  const handleRefresh = () => {
    console.log('Refreshing logs...');
    setFilteredLogs([...logs]);
  };

  const handleClear = () => {
    setLogs([]);
    setFilteredLogs([]);
  };

  const handleExport = () => {
    // Convert logs to CSV format
    const headers = ['Date', 'Time', 'Location', 'Status', 'Severity', 'Response'];
    const csvData = [
      headers.join(','),
      ...filteredLogs.map(log => 
        [log.date, log.time, log.location, log.status, log.severity, log.response].join(',')
      )
    ].join('\n');

    // Create blob and download
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `incident-logs-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log('Logs exported successfully!');
  };

  return (
    <div className="logs-container">
      <Card 
        title={<h2>Incident Logs</h2>}
        className="logs-card"
      >
        {/* Filters and Actions */}
        <div className="logs-header">
          <Space wrap style={{ marginBottom: 16 }}>
            <Input
              placeholder="Search logs..."
              prefix={<SearchOutlined />}
              style={{ width: 200 }}
            />
            <RangePicker />
            <Select defaultValue="all" style={{ width: 150 }}>
              <Option value="all">All Locations</Option>
              <Option value="hallway">Hallway</Option>
              <Option value="room">Rooms</Option>
              <Option value="kitchen">Kitchen</Option>
            </Select>
            <Select defaultValue="all" style={{ width: 150 }}>
              <Option value="all">All Status</Option>
              <Option value="fall">Fall Detected</Option>
              <Option value="normal">Normal</Option>
            </Select>
          </Space>

          <Space>
            <Button 
              icon={<DownloadOutlined />} 
              onClick={handleExport}
              type="primary"
            >
              Export
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={handleRefresh}
            >
              Refresh
            </Button>
            <Button 
              icon={<ClearOutlined />} 
              onClick={handleClear}
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
          pagination={{ 
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} logs`
          }}
        />
      </Card>
    </div>
  );
};

export default Logs;
