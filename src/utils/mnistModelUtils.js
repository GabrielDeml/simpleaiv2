import * as tf from '@tensorflow/tfjs';
import { MODEL_CONFIG, DATASET_CONFIG, TRAINING_CONFIG } from '../constants/mnistConfig.js';
import { Logger } from './logger.js';

/**
 * Utility functions for MNIST model operations
 */

/**
 * Create a neural network model for MNIST digit classification
 * @returns {tf.Sequential} Compiled TensorFlow.js model
 */
export function createMNISTModel() {
  const model = tf.sequential({
    layers: [
      // Input layer with ReLU activation
      tf.layers.dense({
        units: MODEL_CONFIG.HIDDEN_UNITS,
        activation: MODEL_CONFIG.ACTIVATION,
        inputShape: [DATASET_CONFIG.IMAGE_SIZE]
      }),
      
      // Dropout for regularization
      tf.layers.dropout({
        rate: MODEL_CONFIG.DROPOUT_RATE
      }),
      
      // Output layer with softmax for classification
      tf.layers.dense({
        units: DATASET_CONFIG.NUM_CLASSES,
        activation: MODEL_CONFIG.OUTPUT_ACTIVATION
      })
    ]
  });

  // Compile the model
  model.compile({
    optimizer: tf.train[MODEL_CONFIG.OPTIMIZER](),
    loss: MODEL_CONFIG.LOSS,
    metrics: MODEL_CONFIG.METRICS
  });

  return model;
}

/**
 * Train the MNIST model with progress callbacks
 * @param {tf.Sequential} model - The model to train
 * @param {Object} trainData - Training data {images, labels}
 * @param {Function} onEpochEnd - Callback for epoch completion
 * @param {Function} onBatchEnd - Callback for batch completion (optional)
 * @returns {Promise<tf.History>} Training history
 */
export async function trainMNISTModel(model, trainData, onEpochEnd, onBatchEnd = null) {
  if (!model || !trainData) {
    throw new Error('Model and training data are required');
  }

  const startTime = performance.now();
  Logger.info('Starting model training...');

  const callbacks = {
    onEpochEnd: (epoch, logs) => {
      Logger.debug(`Epoch ${epoch + 1}/${TRAINING_CONFIG.EPOCHS}:`, {
        loss: logs.loss?.toFixed(4),
        accuracy: logs.acc?.toFixed(4),
        valLoss: logs.val_loss?.toFixed(4),
        valAccuracy: logs.val_acc?.toFixed(4)
      });
      
      if (onEpochEnd) {
        onEpochEnd(epoch, logs);
      }
    }
  };

  if (onBatchEnd) {
    callbacks.onBatchEnd = onBatchEnd;
  }

  try {
    const history = await model.fit(trainData.images, trainData.labels, {
      batchSize: TRAINING_CONFIG.BATCH_SIZE,
      epochs: TRAINING_CONFIG.EPOCHS,
      validationSplit: TRAINING_CONFIG.VALIDATION_SPLIT,
      callbacks
    });

    const trainingTime = performance.now() - startTime;
    Logger.info('Training completed successfully');
    Logger.performance('Model training', trainingTime);

    return history;
  } catch (error) {
    Logger.error('Training failed:', error);
    throw new Error(`Training failed: ${error.message}`);
  }
}

/**
 * Evaluate model performance on test data
 * @param {tf.Sequential} model - Trained model
 * @param {Object} testData - Test data {images, labels}
 * @returns {Promise<{accuracy: number, loss: number}>} Test metrics
 */
export async function evaluateModel(model, testData) {
  if (!model || !testData) {
    throw new Error('Model and test data are required');
  }

  const startTime = performance.now();
  Logger.info('Evaluating model on test dataset...');

  try {
    // Get predictions
    const predictions = model.predict(testData.images);
    const predictedClasses = predictions.argMax(1);
    const trueClasses = testData.labels.argMax(1);
    
    // Calculate accuracy
    const correctPredictions = tf.equal(predictedClasses, trueClasses);
    const accuracy = correctPredictions.mean();
    const accuracyValue = await accuracy.data();
    
    // Calculate loss
    const loss = tf.losses.softmaxCrossEntropy(testData.labels, predictions);
    const lossValue = await loss.data();
    
    // Clean up tensors
    predictions.dispose();
    predictedClasses.dispose();
    trueClasses.dispose();
    correctPredictions.dispose();
    accuracy.dispose();
    loss.dispose();
    
    const evaluationTime = performance.now() - startTime;
    Logger.info(`Model evaluation completed - Accuracy: ${(accuracyValue[0] * 100).toFixed(2)}%`);
    Logger.performance('Model evaluation', evaluationTime);
    
    return {
      accuracy: accuracyValue[0],
      loss: lossValue[0]
    };
  } catch (error) {
    Logger.error('Model evaluation failed:', error);
    throw new Error(`Model evaluation failed: ${error.message}`);
  }
}

/**
 * Make prediction on a single image
 * @param {tf.Sequential} model - Trained model
 * @param {tf.Tensor} imageData - Single image tensor [1, 784]
 * @returns {Promise<{prediction: number, confidence: number, probabilities: Float32Array}>}
 */
export async function predictSingle(model, imageData) {
  if (!model || !imageData) {
    throw new Error('Model and image data are required');
  }

  try {
    const prediction = model.predict(imageData);
    const probabilities = await prediction.data();
    const predictedClass = await prediction.argMax(1).data();
    const confidence = Math.max(...probabilities);
    
    prediction.dispose();
    
    return {
      prediction: predictedClass[0],
      confidence,
      probabilities
    };
  } catch (error) {
    throw new Error(`Prediction failed: ${error.message}`);
  }
}

/**
 * Get sample predictions for debugging
 * @param {tf.Sequential} model - Trained model
 * @param {Object} testData - Test data
 * @param {number} numSamples - Number of samples to predict
 * @returns {Promise<Array>} Array of prediction results
 */
export async function getSamplePredictions(model, testData, numSamples = 5) {
  const results = [];
  
  for (let i = 0; i < numSamples; i++) {
    const imageSlice = testData.images.slice([i, 0], [1, DATASET_CONFIG.IMAGE_SIZE]);
    const labelSlice = testData.labels.slice([i, 0], [1, DATASET_CONFIG.NUM_CLASSES]);
    
    const trueLabel = await labelSlice.argMax(1).data();
    const predictionResult = await predictSingle(model, imageSlice);
    
    results.push({
      index: i,
      true: trueLabel[0],
      predicted: predictionResult.prediction,
      confidence: predictionResult.confidence,
      correct: trueLabel[0] === predictionResult.prediction
    });
    
    imageSlice.dispose();
    labelSlice.dispose();
  }
  
  return results;
}