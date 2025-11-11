import React, { useRef, useEffect, useState } from 'react';
import { Card, Button, Alert } from 'antd';
import { CameraOutlined, VideoCameraAddOutlined } from '@ant-design/icons';
import './WebcamFeed.css';

const WebcamFeed = () => {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    startWebcam();

    return () => {
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setError(null);
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Unable to access webcam. Please ensure you have granted camera permissions.');
      setIsStreaming(false);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  const toggleWebcam = () => {
    if (isStreaming) {
      stopWebcam();
    } else {
      startWebcam();
    }
  };

  return (
    <div className="webcam-feed-page">
      <Card 
        title={<h2>Live Camera Feed</h2>}
        className="webcam-feed-card"
        extra={
          <Button 
            type={isStreaming ? 'default' : 'primary'}
            icon={isStreaming ? <CameraOutlined /> : <VideoCameraAddOutlined />}
            onClick={toggleWebcam}
          >
            {isStreaming ? 'Stop Camera' : 'Start Camera'}
          </Button>
        }
      >
        {error && (
          <Alert
            message="Camera Error"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <div className="video-container">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="webcam-video-feed"
          />
          
          {!isStreaming && !error && (
            <div className="video-placeholder">
              <CameraOutlined style={{ fontSize: 64, color: '#ccc' }} />
              <p>Camera is off. Click "Start Camera" to begin.</p>
            </div>
          )}
        </div>

        <div className="webcam-info">
          <p><strong>Status:</strong> {isStreaming ? '🟢 Live' : '🔴 Offline'}</p>
          <p><strong>Resolution:</strong> 1280x720</p>
          <p><strong>Detection:</strong> MediaPipe Pose Detection Active</p>
        </div>
      </Card>
    </div>
  );
};

export default WebcamFeed;
