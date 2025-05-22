import React, { useEffect, useRef } from 'react';

/**
 * Component for displaying training logs and model metrics
 * @param {Object} props - Component props
 * @param {Array} props.logs - Array of log messages to display
 * @param {Object} props.testResults - Test results if available
 * @param {boolean} props.isTraining - Whether training is in progress
 */
function TrainingLogs({ logs, testResults, isTraining }) {
  const logsEndRef = useRef(null);

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Format training log entry
  const formatLogEntry = (log, index) => {
    if (typeof log === 'string') {
      return log;
    }
    
    // If log is an object (structured log entry)
    if (log.epoch) {
      return `Epoch ${log.epoch}: loss=${log.loss?.toFixed(4) || 'N/A'}, acc=${log.accuracy?.toFixed(4) || 'N/A'}, val_loss=${log.valLoss?.toFixed(4) || 'N/A'}, val_acc=${log.valAccuracy?.toFixed(4) || 'N/A'}`;
    }
    
    return JSON.stringify(log);
  };

  // Create status message
  const getStatusMessage = () => {
    if (isTraining) {
      return 'Training in progress...';
    }
    
    if (testResults) {
      return `Test completed - Accuracy: ${(testResults.accuracy * 100).toFixed(2)}%, Loss: ${testResults.loss?.toFixed(4) || 'N/A'}`;
    }
    
    if (logs.length === 0) {
      return 'No logs yet. Click "Train Model" to start training!';
    }
    
    return null;
  };

  const statusMessage = getStatusMessage();

  return (
    <div className="logs-section">
      <h2 className="logs-title">Training Logs</h2>
      <div className="logs-container" role="log" aria-live="polite">
        {statusMessage && (
          <div 
            className={`status-message ${isTraining ? 'training' : ''}`}
            style={{ 
              textAlign: 'center', 
              color: isTraining ? '#4facfe' : '#a0aec0',
              fontStyle: 'italic',
              marginBottom: logs.length > 0 ? '1rem' : '0'
            }}
          >
            {statusMessage}
          </div>
        )}
        
        {logs.map((log, index) => (
          <div key={index} className="log-entry">
            {formatLogEntry(log, index)}
          </div>
        ))}
        
        {/* Test results display */}
        {testResults && (
          <div className="log-entry test-result">
            <strong>
              Final Test Results: Accuracy {(testResults.accuracy * 100).toFixed(2)}%
              {testResults.loss && `, Loss ${testResults.loss.toFixed(4)}`}
            </strong>
          </div>
        )}
        
        <div ref={logsEndRef} />
      </div>
    </div>
  );
}

export default TrainingLogs;