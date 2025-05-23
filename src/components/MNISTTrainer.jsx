import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { MnistData } from '../utils/mnistData.js';
import { createMNISTModel, modelConfig } from '../config/modelConfig.js';
import './MNISTTrainer.css';

const MNISTTrainer = () => {
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainLoss, setTrainLoss] = useState(0);
  const [trainAccuracy, setTrainAccuracy] = useState(0);
  const [epochs, setEpochs] = useState(10);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [mnistData, setMnistData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [testAccuracy, setTestAccuracy] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);


  // Load real MNIST data using our custom loader
  const loadMNISTData = async () => {
    setIsLoading(true);
    setLoadingStatus('Loading MNIST dataset...');
    try {
      const data = new MnistData();
      await data.load();
      
      setLoadingStatus('Processing data...');
      
      const { xs, ys } = data.getTrainData(modelConfig.trainingData.numExamples);
      const testData = data.getTestData(modelConfig.trainingData.testNumExamples);
      
      setMnistData({
        train: {
          images: xs,
          labels: ys
        },
        test: {
          images: testData.xs,
          labels: testData.ys
        },
        dataLoader: data
      });
      
      setLoadingStatus('Data loaded successfully!');
      setTimeout(() => setLoadingStatus(''), 2000);
    } catch (error) {
      console.error('Error loading MNIST data:', error);
      setLoadingStatus('Failed to load data');
      alert('Failed to load MNIST data. Please check your internet connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };


  // Train the model
  const trainModel = async () => {
    if (!model || !mnistData) return;
    
    setIsTraining(true);
    setCurrentEpoch(0);
    
    try {
      await model.fit(mnistData.train.images, mnistData.train.labels, {
        epochs: epochs,
        ...modelConfig.training,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setCurrentEpoch(epoch + 1);
            setTrainLoss(logs.loss.toFixed(4));
            setTrainAccuracy((logs.acc * 100).toFixed(2));
          }
        }
      });
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setIsTraining(false);
    }
  };

  // Evaluate model on test dataset
  const evaluateOnTestSet = async () => {
    if (!model || !mnistData || !mnistData.test) {
      console.log('Model or test data not ready');
      return;
    }

    setIsEvaluating(true);
    try {
      // Evaluate the model on the test set
      const result = await model.evaluate(mnistData.test.images, mnistData.test.labels);
      const testAcc = result[1].dataSync()[0];
      
      setTestAccuracy((testAcc * 100).toFixed(2));
      result[0].dispose();
      result[1].dispose();
    } catch (error) {
      console.error('Evaluation error:', error);
    } finally {
      setIsEvaluating(false);
    }
  };


  // Canvas drawing functions
  const startDrawing = (e) => {
    isDrawing.current = true;
    draw(e);
  };

  const stopDrawing = () => {
    isDrawing.current = false;
  };

  const draw = (e) => {
    if (!isDrawing.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  };

  const predictDigit = async () => {
    if (!model || !canvasRef.current) return;
    
    if (currentEpoch === 0) {
      alert('Please train the model first before making predictions!');
      return;
    }
    
    const canvas = canvasRef.current;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    tempCtx.filter = 'invert(1)';
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    const processedData = new Float32Array(28 * 28);
    
    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      const grayValue = data[pixelIndex];
      processedData[i] = (255 - grayValue) / 255.0;
    }
    
    const input = tf.tensor4d(processedData, [1, 28, 28, 1]);
    
    const output = await model.predict(input).data();
    const maxIndex = output.indexOf(Math.max(...output));
    setPrediction(maxIndex);
    
    input.dispose();
  };

  // Initialize canvas
  useEffect(() => {
    if (canvasRef.current) {
      clearCanvas();
    }
  }, []);

  // Initialize model
  useEffect(() => {
    const newModel = createMNISTModel();
    setModel(newModel);
  }, []);

  return (
    <div className="mnist-trainer">
      <h1>MNIST Digit Recognition</h1>
      
      <div className="trainer-container">
        <div className="controls-section">
          <h3>Training Controls</h3>
          
          <div className="control-group">
            <button 
              onClick={loadMNISTData} 
              disabled={isLoading || isTraining || mnistData !== null}
              className="btn-primary"
            >
              {isLoading ? 'Loading...' : mnistData ? 'Data Loaded ✓' : 'Load MNIST Data'}
            </button>
            {loadingStatus && <p style={{fontSize: '14px', marginTop: '10px', color: '#4CAF50'}}>{loadingStatus}</p>}
          </div>
          
          <div className="control-group">
            <label>
              Epochs: {epochs}
              <input 
                type="range" 
                min="1" 
                max="30" 
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                disabled={isTraining}
              />
            </label>
          </div>
          
          <div className="control-group">
            <button 
              onClick={trainModel}
              disabled={!mnistData || isTraining || !model}
            >
              {isTraining ? `Training... (${currentEpoch}/${epochs})` : 'Train Model'}
            </button>
          </div>
          
          <div className="control-group">
            <button 
              onClick={evaluateOnTestSet}
              disabled={!mnistData || !model || currentEpoch === 0 || isEvaluating}
              className="btn-secondary"
            >
              {isEvaluating ? 'Evaluating...' : 'Evaluate on Test Set'}
            </button>
          </div>
          
          <div className="metrics">
            <h4>Training Metrics</h4>
            <p>Loss: {trainLoss}</p>
            <p>Training Accuracy: {trainAccuracy}%</p>
            {testAccuracy !== null && <p>Test Accuracy: {testAccuracy}%</p>}
            <p>Epoch: {currentEpoch}/{epochs}</p>
          </div>
        </div>
        
        <div className="canvas-section">
          <h3>Draw a Digit</h3>
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseUp={stopDrawing}
            onMouseMove={draw}
            onMouseLeave={stopDrawing}
          />
          
          <div className="canvas-controls">
            <button onClick={clearCanvas}>Clear</button>
            <button onClick={predictDigit} disabled={!model || isTraining}>
              Predict
            </button>
          </div>
          
          {prediction !== null && (
            <div className="prediction">
              <h2>Prediction: {prediction}</h2>
            </div>
          )}
        </div>
      </div>
      
      <div className="model-info">
        <h3>Model Architecture</h3>
        <ul>
          <li>Conv2D (32 filters, 3x3) + ReLU</li>
          <li>MaxPooling2D (2x2)</li>
          <li>Conv2D (64 filters, 3x3) + ReLU</li>
          <li>MaxPooling2D (2x2)</li>
          <li>Flatten</li>
          <li>Dropout (0.2)</li>
          <li>Dense (128) + ReLU</li>
          <li>Dense (10) + Softmax</li>
        </ul>
      </div>
    </div>
  );
};

export default MNISTTrainer;