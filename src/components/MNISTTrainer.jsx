import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { MnistData } from '../utils/mnistData';
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

  // Create CNN model
  const createModel = () => {
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [28, 28, 1],
          kernelSize: 3,
          filters: 32,
          activation: 'relu'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({
          kernelSize: 3,
          filters: 64,
          activation: 'relu'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.flatten(),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({
          units: 128,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 10,
          activation: 'softmax'
        })
      ]
    });

    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  };

  // Load real MNIST data using our custom loader
  const loadMNISTData = async () => {
    setIsLoading(true);
    setLoadingStatus('Loading MNIST dataset...');
    try {
      console.log('Loading MNIST data...');
      
      // Create and load MNIST data
      const data = new MnistData();
      await data.load();
      
      setLoadingStatus('Processing data...');
      
      // Get a subset of training data for faster training
      const numExamples = 5000;
      const { xs, ys } = data.getTrainData(numExamples);
      
      // Also get test data
      const testNumExamples = 1000;
      const testData = data.getTestData(testNumExamples);
      
      setMnistData({
        train: {
          images: xs,
          labels: ys
        },
        test: {
          images: testData.xs,
          labels: testData.ys
        },
        dataLoader: data  // Keep reference to data loader
      });
      
      console.log('MNIST data loaded successfully');
      console.log(`Training data shape: ${xs.shape}`);
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

  // Remove unused function
  // loadMNISTDataSimple has been replaced with loadMNISTData

  // Train the model
  const trainModel = async () => {
    if (!model || !mnistData) return;
    
    setIsTraining(true);
    setCurrentEpoch(0);
    
    try {
      await model.fit(mnistData.train.images, mnistData.train.labels, {
        epochs: epochs,
        batchSize: 128,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setCurrentEpoch(epoch + 1);
            setTrainLoss(logs.loss.toFixed(4));
            setTrainAccuracy((logs.acc * 100).toFixed(2));
            console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Accuracy: ${(logs.acc * 100).toFixed(2)}%`);
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
      const testLoss = result[0].dataSync()[0];
      const testAcc = result[1].dataSync()[0];
      
      setTestAccuracy((testAcc * 100).toFixed(2));
      console.log(`Test Loss: ${testLoss.toFixed(4)}, Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
      
      // Clean up tensors
      result[0].dispose();
      result[1].dispose();
      
      // Test a few individual predictions
      console.log('\n=== Sample Test Predictions ===');
      for (let i = 0; i < 5; i++) {
        const testImage = mnistData.test.images.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const testLabel = mnistData.test.labels.slice([i, 0], [1, 10]);
        
        const labelData = await testLabel.data();
        const actualDigit = labelData.indexOf(1);
        
        const prediction = await model.predict(testImage).data();
        const predictedDigit = prediction.indexOf(Math.max(...prediction));
        
        console.log(`Sample ${i + 1}: Actual: ${actualDigit}, Predicted: ${predictedDigit}, Correct: ${actualDigit === predictedDigit ? '✓' : '✗'}`);
        
        testImage.dispose();
        testLabel.dispose();
      }
    } catch (error) {
      console.error('Evaluation error:', error);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Test prediction with actual MNIST data
  const testPredictionWithMNIST = async () => {
    if (!model || !mnistData) {
      console.log('Model or data not ready');
      return;
    }

    // Test multiple samples from BOTH train and test sets
    console.log('\n=== Testing on TRAINING samples ===');
    for (let i = 0; i < 3; i++) {
      // Get a single test image from the training data
      const testImage = mnistData.train.images.slice([i, 0, 0, 0], [1, 28, 28, 1]);
      const testLabel = mnistData.train.labels.slice([i, 0], [1, 10]);
      
      // Get the actual label
      const labelData = await testLabel.data();
      const actualDigit = labelData.indexOf(1);
      
      // Make prediction
      const prediction = await model.predict(testImage).data();
      const predictedDigit = prediction.indexOf(Math.max(...prediction));
      
      console.log(`\n=== MNIST Test Prediction #${i + 1} ===`);
      console.log('Label one-hot:', Array.from(labelData));
      console.log('Actual digit:', actualDigit);
      console.log('Predicted digit:', predictedDigit);
      console.log('Prediction probabilities:', Array.from(prediction).map((p, i) => `${i}: ${(p * 100).toFixed(2)}%`));
      
      testImage.dispose();
      testLabel.dispose();
    }
    
    // Now test on TEST samples
    console.log('\n=== Testing on TEST samples ===');
    for (let i = 0; i < 3; i++) {
      const testImage = mnistData.test.images.slice([i, 0, 0, 0], [1, 28, 28, 1]);
      const testLabel = mnistData.test.labels.slice([i, 0], [1, 10]);
      
      const labelData = await testLabel.data();
      const actualDigit = labelData.indexOf(1);
      
      const prediction = await model.predict(testImage).data();
      const predictedDigit = prediction.indexOf(Math.max(...prediction));
      
      console.log(`Test Sample ${i + 1}: Actual: ${actualDigit}, Predicted: ${predictedDigit}, Correct: ${actualDigit === predictedDigit ? '✓' : '✗'}`);
      console.log('Prediction probabilities:', Array.from(prediction).map((p, i) => `${i}: ${(p * 100).toFixed(2)}%`));
      
      testImage.dispose();
      testLabel.dispose();
    }
    
    // Check data statistics
    console.log('\n=== Data Statistics ===');
    const trainSample = await mnistData.train.images.slice([0, 0, 0, 0], [1, 28, 28, 1]).data();
    const testSample = await mnistData.test.images.slice([0, 0, 0, 0], [1, 28, 28, 1]).data();
    
    console.log('Train sample stats - Min:', Math.min(...trainSample), 'Max:', Math.max(...trainSample), 'Mean:', trainSample.reduce((a,b) => a+b) / trainSample.length);
    console.log('Test sample stats - Min:', Math.min(...testSample), 'Max:', Math.max(...testSample), 'Mean:', testSample.reduce((a,b) => a+b) / testSample.length);
    
    // Visualize first train and test image
    console.log('\n=== Visual Check ===');
    const visualizeImage = (data, label) => {
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(28, 28);
      
      for (let i = 0; i < 28 * 28; i++) {
        const value = Math.floor(data[i] * 255);
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
      }
      
      ctx.putImageData(imageData, 0, 0);
      console.log(`Image (label ${label}):`, canvas.toDataURL());
    };
    
    const trainLabel = await mnistData.train.labels.slice([0, 0], [1, 10]).data();
    const trainDigit = trainLabel.indexOf(1);
    visualizeImage(trainSample, trainDigit);
    
    const testLabel = await mnistData.test.labels.slice([0, 0], [1, 10]).data();
    const testDigit = testLabel.indexOf(1);
    visualizeImage(testSample, testDigit);
    
    // Also check model weights
    console.log('\n=== Model Information ===');
    const weights = model.getWeights();
    console.log('Number of weight tensors:', weights.length);
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      const data = await w.data();
      console.log(`Weight ${i} shape:`, w.shape);
      // Use reduce for large arrays to avoid stack overflow
      const min = data.reduce((a, b) => Math.min(a, b));
      const max = data.reduce((a, b) => Math.max(a, b));
      const mean = data.reduce((a, b) => a + b) / data.length;
      console.log(`Weight ${i} stats - Min: ${min.toFixed(4)}, Max: ${max.toFixed(4)}, Mean: ${mean.toFixed(4)}`);
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
    
    // Check if model has been trained
    if (currentEpoch === 0) {
      alert('Please train the model first before making predictions!');
      return;
    }
    
    const canvas = canvasRef.current;
    
    // Get image data and resize to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with white background first (like MNIST)
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    // Draw the canvas content scaled down with black color
    tempCtx.filter = 'invert(1)';
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Create a Float32Array for the processed image
    const processedData = new Float32Array(28 * 28);
    
    // Process the image: convert to grayscale and normalize
    // MNIST data is normalized between 0 and 1, where 0 is white and 1 is black
    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      // Get grayscale value (using red channel since it's already grayscale)
      const grayValue = data[pixelIndex];
      // Normalize: white (255) becomes 0, black (0) becomes 1
      processedData[i] = (255 - grayValue) / 255.0;
    }
    
    // Create tensor from processed data
    const input = tf.tensor4d(processedData, [1, 28, 28, 1]);
    
    // Debug: log input tensor stats
    console.log('Input tensor stats:');
    console.log('Min:', Math.min(...processedData));
    console.log('Max:', Math.max(...processedData));
    console.log('Mean:', processedData.reduce((a, b) => a + b) / processedData.length);
    console.log('Non-zero pixels:', processedData.filter(x => x > 0.1).length);
    
    // Visual debug: show the processed image
    const debugCanvas = document.createElement('canvas');
    debugCanvas.width = 28;
    debugCanvas.height = 28;
    const debugCtx = debugCanvas.getContext('2d');
    const debugImageData = debugCtx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
      const value = Math.floor(processedData[i] * 255);
      debugImageData.data[i * 4] = value;
      debugImageData.data[i * 4 + 1] = value;
      debugImageData.data[i * 4 + 2] = value;
      debugImageData.data[i * 4 + 3] = 255;
    }
    debugCtx.putImageData(debugImageData, 0, 0);
    console.log('Processed image preview:', debugCanvas.toDataURL());
    
    // Make prediction
    const output = await model.predict(input).data();
    const maxIndex = output.indexOf(Math.max(...output));
    setPrediction(maxIndex);
    
    // Log prediction probabilities for debugging
    console.log('Prediction probabilities:', Array.from(output).map((p, i) => `${i}: ${(p * 100).toFixed(2)}%`));
    console.log('Predicted digit:', maxIndex);
    
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
    const newModel = createModel();
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
          
          <div className="control-group">
            <button 
              onClick={testPredictionWithMNIST}
              disabled={!mnistData || !model || currentEpoch === 0}
              style={{fontSize: '12px', padding: '5px 10px'}}
            >
              Debug MNIST Samples
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