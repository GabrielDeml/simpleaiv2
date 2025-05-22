import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import './MNISTTrainer.css';

const MNISTTrainer = () => {
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainData, setTrainData] = useState(null);
  const [testData, setTestData] = useState(null);
  const [logs, setLogs] = useState([]);
  const canvasRef = useRef(null);

  // Load MNIST data
  const loadData = async () => {
    console.log('Loading MNIST data...');
    const IMAGE_SIZE = 784;
    const NUM_CLASSES = 10;
    const NUM_DATASET_ELEMENTS = 65000;
    const TRAIN_TEST_RATIO = 5/6;
    const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);

    const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

    // Load images
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    await new Promise((resolve) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
        
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize);
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }

        const datasetImages = new Float32Array(datasetBytesBuffer);
        resolve(datasetImages);
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    }).then(async (datasetImages) => {
      // Load labels - this file contains pre-encoded one-hot vectors (10 bytes per label)
      const labelsResponse = await fetch(MNIST_LABELS_PATH);
      const labelsBuffer = await labelsResponse.arrayBuffer();
      // Skip any header and read the one-hot encoded data
      const labelsData = new Uint8Array(labelsBuffer);
      
      // The labels are already one-hot encoded as 10 bytes per label
      // So we need to reshape them correctly
      const labelsArray = new Float32Array(NUM_DATASET_ELEMENTS * NUM_CLASSES);
      for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
        for (let j = 0; j < NUM_CLASSES; j++) {
          labelsArray[i * NUM_CLASSES + j] = labelsData[i * NUM_CLASSES + j];
        }
      }

      // Split into train and test
      const trainImages = datasetImages.slice(0, NUM_TRAIN_ELEMENTS * IMAGE_SIZE);
      const testImages = datasetImages.slice(NUM_TRAIN_ELEMENTS * IMAGE_SIZE, NUM_DATASET_ELEMENTS * IMAGE_SIZE);
      const trainLabels = labelsArray.slice(0, NUM_TRAIN_ELEMENTS * NUM_CLASSES);
      const testLabels = labelsArray.slice(NUM_TRAIN_ELEMENTS * NUM_CLASSES);

      const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
      
      console.log('=== DATA LOADING DEBUG ===');
      console.log('NUM_DATASET_ELEMENTS:', NUM_DATASET_ELEMENTS);
      console.log('NUM_TRAIN_ELEMENTS:', NUM_TRAIN_ELEMENTS);
      console.log('NUM_TEST_ELEMENTS:', NUM_TEST_ELEMENTS);
      console.log('IMAGE_SIZE:', IMAGE_SIZE);
      console.log('datasetImages.length:', datasetImages.length);
      console.log('labelsData.length:', labelsData.length);
      console.log('trainImages.length:', trainImages.length);
      console.log('testImages.length:', testImages.length);
      console.log('trainLabels.length:', trainLabels.length);
      console.log('testLabels.length:', testLabels.length);
      console.log('Expected train images length:', NUM_TRAIN_ELEMENTS * IMAGE_SIZE);
      console.log('Expected test images length:', NUM_TEST_ELEMENTS * IMAGE_SIZE);
      
      // Sample some labels to verify they look correct (first few one-hot vectors)
      console.log('First train label vector:', Array.from(trainLabels.slice(0, 10)));
      console.log('Second train label vector:', Array.from(trainLabels.slice(10, 20)));
      console.log('First test label vector:', Array.from(testLabels.slice(0, 10)));
      
      const trainImagesTensor = tf.tensor2d(trainImages, [NUM_TRAIN_ELEMENTS, IMAGE_SIZE]);
      const trainLabelsTensor = tf.tensor2d(trainLabels, [NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
      
      console.log('trainImagesTensor.shape:', trainImagesTensor.shape);
      console.log('trainLabelsTensor.shape:', trainLabelsTensor.shape);
      
      setTrainData({
        images: trainImagesTensor,
        labels: trainLabelsTensor
      });

      const testImagesTensor = tf.tensor2d(testImages, [NUM_TEST_ELEMENTS, IMAGE_SIZE]);
      const testLabelsTensor = tf.tensor2d(testLabels, [NUM_TEST_ELEMENTS, NUM_CLASSES]);
      
      console.log('testImagesTensor.shape:', testImagesTensor.shape);
      console.log('testLabelsTensor.shape:', testLabelsTensor.shape);
      
      setTestData({
        images: testImagesTensor,
        labels: testLabelsTensor
      });

      console.log('=== MNIST DATA LOADED SUCCESSFULLY ===');
    });
  };

  // Create model
  const createModel = () => {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 10, activation: 'softmax' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  };

  // Train model
  const trainModel = async () => {
    if (!trainData || !testData) {
      alert('Please load data first');
      return;
    }

    setIsTraining(true);
    const newModel = createModel();
    setModel(newModel);

    const batchSize = 512;
    const epochs = 10;

    const history = await newModel.fit(trainData.images, trainData.labels, {
      batchSize,
      epochs,
      validationSplit: 0.15,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setLogs(prev => [...prev, `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${logs.val_acc.toFixed(4)}`]);
        }
      }
    });

    setIsTraining(false);
    console.log('Training completed');
  };

  // Test model
  const testModel = async () => {
    if (!model || !testData) {
      alert('Please train model first');
      return;
    }

    console.log('=== TESTING MODEL ===');
    console.log('testData.images.shape:', testData.images.shape);
    console.log('testData.labels.shape:', testData.labels.shape);

    try {
      const predictions = model.predict(testData.images);
      console.log('predictions.shape:', predictions.shape);
      
      const predictedClasses = predictions.argMax(1);
      const trueClasses = testData.labels.argMax(1);
      
      console.log('predictedClasses.shape:', predictedClasses.shape);
      console.log('trueClasses.shape:', trueClasses.shape);
      
      // Sample first few predictions vs actual for verification
      const samplePredicted = await predictedClasses.slice([0], [5]).data();
      const sampleTrue = await trueClasses.slice([0], [5]).data();
      console.log('Sample predicted:', Array.from(samplePredicted));
      console.log('Sample actual:', Array.from(sampleTrue));
      
      const correctPredictions = tf.equal(predictedClasses, trueClasses);
      const accuracy = correctPredictions.mean();
      const accuracyValue = await accuracy.data();
      
      setLogs(prev => [...prev, `Test Accuracy: ${(accuracyValue[0] * 100).toFixed(2)}%`]);
      
      // Clean up tensors
      predictions.dispose();
      predictedClasses.dispose();
      trueClasses.dispose();
      correctPredictions.dispose();
      accuracy.dispose();
      
      console.log('=== MODEL TEST COMPLETED ===');
    } catch (error) {
      console.error('Error during model testing:', error);
      setLogs(prev => [...prev, `Error during testing: ${error.message}`]);
    }
  };

  // Draw random test image
  const drawRandomImage = async () => {
    if (!testData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    const randomIndex = Math.floor(Math.random() * testData.images.shape[0]);
    
    console.log('=== DEBUGGING RANDOM IMAGE ===');
    console.log('randomIndex:', randomIndex);
    console.log('testData.images.shape:', testData.images.shape);
    console.log('testData.labels.shape:', testData.labels.shape);
    
    // Get image data
    const imageData = await testData.images.slice([randomIndex, 0], [1, 784]).data();
    
    // Get label data - let's examine the raw one-hot vector first
    const labelTensor = testData.labels.slice([randomIndex, 0], [1, 10]);
    const labelVector = await labelTensor.data();
    console.log('Raw label vector:', Array.from(labelVector));
    
    const actualLabel = await labelTensor.argMax(1).data();
    console.log('Extracted actual label:', actualLabel[0]);

    // Draw image
    const imageArray = new Uint8ClampedArray(28 * 28 * 4);
    for (let i = 0; i < 28 * 28; i++) {
      const pixelValue = imageData[i] * 255;
      imageArray[i * 4] = pixelValue;     // R
      imageArray[i * 4 + 1] = pixelValue; // G
      imageArray[i * 4 + 2] = pixelValue; // B
      imageArray[i * 4 + 3] = 255;       // A
    }

    const imgData = new ImageData(imageArray, 28, 28);
    ctx.putImageData(imgData, 0, 0);

    // Predict if model exists
    if (model) {
      const predictionTensor = model.predict(testData.images.slice([randomIndex, 0], [1, 784]));
      const predictionVector = await predictionTensor.data();
      console.log('Raw prediction vector:', Array.from(predictionVector));
      
      const predictedClass = await predictionTensor.argMax(1).data();
      console.log('Extracted predicted class:', predictedClass[0]);
      
      setLogs(prev => [...prev, `Actual: ${actualLabel[0]}, Predicted: ${predictedClass[0]}`]);
      
      predictionTensor.dispose();
    }
    
    labelTensor.dispose();
    console.log('=== END DEBUGGING ===');
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="mnist-container">
      <div className="mnist-card">
        <h1 className="mnist-title">MNIST Digit Recognition Trainer</h1>
        
        <div className="controls-section">
          <button 
            onClick={trainModel} 
            disabled={isTraining || !trainData}
            className="control-button train-button"
          >
            {isTraining && <span className="loading-indicator"></span>}
            {isTraining ? 'Training...' : 'Train Model'}
          </button>
          
          <button 
            onClick={testModel} 
            disabled={!model}
            className="control-button test-button"
          >
            Test Model
          </button>
          
          <button 
            onClick={drawRandomImage} 
            disabled={!testData}
            className="control-button draw-button"
          >
            Draw Random Test Image
          </button>
        </div>

        <div className="content-section">
          <div className="image-section">
            <h2 className="section-title">Test Image</h2>
            <div className="canvas-container">
              <canvas 
                ref={canvasRef} 
                width={28} 
                height={28} 
                className="mnist-canvas"
                style={{ 
                  width: '200px',
                  height: '200px'
                }}
              />
            </div>
          </div>
          
          <div className="logs-section">
            <h2 className="logs-title">Training Logs</h2>
            <div className="logs-container">
              {logs.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#a0aec0', fontStyle: 'italic' }}>
                  No logs yet. Click "Train Model" to start training!
                </div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className="log-entry">{log}</div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MNISTTrainer;