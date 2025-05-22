import React, { useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

// Custom hooks
import { useMNISTData } from '../hooks/useMNISTData.js';
import { useMNISTModel } from '../hooks/useMNISTModel.js';

// Components
import ControlPanel from './ControlPanel.jsx';
import ImageCanvas from './ImageCanvas.jsx';
import TrainingLogs from './TrainingLogs.jsx';
import ErrorBoundary from './ErrorBoundary.jsx';
import ModelBuilder from './ModelBuilder/ModelBuilder.jsx';

// Styles
import './MNISTTrainer.css';

/**
 * Main MNIST Trainer Component
 * Provides a complete interface for training and testing MNIST digit recognition models
 */
function MNISTTrainer() {
  // Custom hooks for data and model management
  const {
    trainData,
    testData,
    isLoading: isDataLoading,
    error: dataError,
    getRandomTestSample,
    dataStats
  } = useMNISTData();

  const {
    model,
    isTraining,
    isTesting,
    trainingHistory,
    testResults,
    error: modelError,
    trainingProgress,
    modelStats,
    createModel,
    trainModel,
    testModel,
    predict,
    clearError
  } = useMNISTModel();

  // Local state for UI interactions
  const [logs, setLogs] = useState([]);
  const [currentSample, setCurrentSample] = useState(null);
  const [currentArchitecture, setCurrentArchitecture] = useState(null);
  const [architectureValid, setArchitectureValid] = useState(false);
  
  // Canvas reference
  const canvasRef = useRef(null);

  /**
   * Add a log message to the logs array
   */
  const addLog = useCallback((message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  }, []);

  /**
   * Handle model architecture changes from ModelBuilder
   */
  const handleModelChange = useCallback((layers, isValid) => {
    setCurrentArchitecture(layers);
    setArchitectureValid(isValid);
    
    if (isValid && layers.length > 1) {
      addLog(`Architecture updated: ${layers.length} layers, ${layers.filter(l => l.type === 'dense').length} dense layers`);
    }
  }, [addLog]);

  /**
   * Handle model training
   */
  const handleTrain = useCallback(async () => {
    if (!trainData) {
      addLog('Error: Training data not available');
      return;
    }

    // Create model with current architecture if available
    if (currentArchitecture && architectureValid && !model) {
      try {
        addLog('Creating model from architecture...');
        createModel(currentArchitecture);
        addLog('Model created successfully!');
      } catch (error) {
        addLog(`Failed to create model: ${error.message}`);
        return;
      }
    }

    try {
      clearError();
      addLog('Starting model training...');
      
      await trainModel(trainData);
      
      addLog('Training completed successfully!');
      
      // Display final training metrics
      if (trainingHistory.length > 0) {
        const lastEpoch = trainingHistory[trainingHistory.length - 1];
        addLog(`Final training accuracy: ${(lastEpoch.accuracy * 100).toFixed(2)}%`);
        addLog(`Final validation accuracy: ${(lastEpoch.valAccuracy * 100).toFixed(2)}%`);
      }
      
    } catch (error) {
      addLog(`Training failed: ${error.message}`);
    }
  }, [trainData, trainModel, trainingHistory, addLog, clearError, currentArchitecture, architectureValid, model, createModel]);

  /**
   * Handle model testing
   */
  const handleTest = useCallback(async () => {
    if (!model || !testData) {
      addLog('Error: Model or test data not available');
      return;
    }

    try {
      clearError();
      addLog('Testing model on test dataset...');
      
      const results = await testModel(testData);
      
      addLog(`Test completed - Accuracy: ${(results.accuracy * 100).toFixed(2)}%`);
      
    } catch (error) {
      addLog(`Testing failed: ${error.message}`);
    }
  }, [model, testData, testModel, addLog, clearError]);

  /**
   * Handle drawing random test image with prediction
   */
  const handleDrawRandom = useCallback(async () => {
    if (!testData || !canvasRef.current) {
      addLog('Error: Test data or canvas not available');
      return;
    }

    try {
      // Get random sample
      const sample = await getRandomTestSample();
      setCurrentSample(sample);
      
      // Draw image on canvas
      canvasRef.current.drawImage(sample.imageData);
      
      let logMessage = `Showing digit: ${sample.label}`;
      
      // Make prediction if model is available
      if (model) {
        const imageSlice = testData.images.slice([sample.index, 0], [1, 784]);
        const prediction = await predict(imageSlice);
        
        const isCorrect = prediction.prediction === sample.label;
        const confidence = (prediction.confidence * 100).toFixed(1);
        
        logMessage += `, Predicted: ${prediction.prediction} (${confidence}% confidence)`;
        logMessage += isCorrect ? ' ✓' : ' ✗';
        
        // Clean up tensor
        imageSlice.dispose();
      }
      
      addLog(logMessage);
      
    } catch (error) {
      addLog(`Failed to draw random image: ${error.message}`);
    }
  }, [testData, model, getRandomTestSample, predict, addLog]);

  /**
   * Get current error message
   */
  const getCurrentError = () => {
    return dataError || modelError;
  };

  /**
   * Handle error dismissal
   */
  const handleDismissError = useCallback(() => {
    clearError();
  }, [clearError]);

  // Determine component states
  const canTrain = !!(trainData && !isTraining && (model || (currentArchitecture && architectureValid)));
  const canTest = !!(model && testData && !isTesting);
  const canDraw = !!(testData && !isDataLoading);
  const hasError = !!(dataError || modelError);

  return (
    <ErrorBoundary>
      <div className="mnist-container">
        <div className="mnist-card">
          <h1 className="mnist-title">MNIST Digit Recognition Trainer</h1>
          
          {/* Error Display */}
          {hasError && (
            <div className="error-banner">
              <span className="error-text">{getCurrentError()}</span>
              <button 
                onClick={handleDismissError}
                className="error-dismiss"
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}
          
          {/* Data Loading Status */}
          {isDataLoading && (
            <div className="loading-banner">
              <span className="loading-indicator"></span>
              Loading MNIST dataset...
            </div>
          )}
          
          {/* Model Builder */}
          <ModelBuilder
            onModelChange={handleModelChange}
          />

          {/* Control Panel */}
          <ControlPanel
            onTrain={handleTrain}
            onTest={handleTest}
            onDrawRandom={handleDrawRandom}
            isTraining={isTraining}
            canTrain={canTrain}
            canTest={canTest}
            canDraw={canDraw}
            trainingProgress={trainingProgress}
          />

          <div className="content-section">
            {/* Image Display */}
            <ImageCanvas ref={canvasRef} />
            
            {/* Training Logs */}
            <TrainingLogs 
              logs={logs}
              testResults={testResults}
              isTraining={isTraining}
            />
          </div>
          
          {/* Model Statistics */}
          {(dataStats.isLoaded || modelStats.isReady) && (
            <div className="stats-section">
              <div className="stats-grid">
                {dataStats.isLoaded && (
                  <div className="stat-item">
                    <span className="stat-label">Training Samples:</span>
                    <span className="stat-value">{dataStats.trainSize.toLocaleString()}</span>
                  </div>
                )}
                {dataStats.isLoaded && (
                  <div className="stat-item">
                    <span className="stat-label">Test Samples:</span>
                    <span className="stat-value">{dataStats.testSize.toLocaleString()}</span>
                  </div>
                )}
                {modelStats.lastTrainingAccuracy && (
                  <div className="stat-item">
                    <span className="stat-label">Training Accuracy:</span>
                    <span className="stat-value">
                      {(modelStats.lastTrainingAccuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
                {modelStats.testAccuracy && (
                  <div className="stat-item">
                    <span className="stat-label">Test Accuracy:</span>
                    <span className="stat-value">
                      {(modelStats.testAccuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default MNISTTrainer;