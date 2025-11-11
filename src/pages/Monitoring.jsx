import React from 'react';
import { Card, Row, Col, Empty } from 'antd';
import { MonitorOutlined } from '@ant-design/icons';
import './Monitoring.css';

const Monitoring = () => {
  return (
    <div className="monitoring-container">
      <Card title={<h2>Monitoring</h2>} className="monitoring-card">
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <div className="placeholder-content">
              <Empty
                image={<MonitorOutlined style={{ fontSize: 64, color: '#1890ff' }} />}
                description={
                  <div>
                    <h3>Monitoring Page</h3>
                    <p>This page is under construction and will be updated soon.</p>
                    <p>Here you'll be able to monitor multiple camera feeds simultaneously.</p>
                  </div>
                }
              />
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default Monitoring;
