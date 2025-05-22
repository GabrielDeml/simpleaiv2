import { useState, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createMNISTModel, trainMNISTModel, evaluateModel, predictSingle } from '../utils/mnistModelUtils.js';
import { createDynamicModel } from '../utils/dynamicModelUtils.js';

/**
 * Custom hook for managing MNIST model training and inference
 * @returns {Object} Model state and methods
 */
export function useMNISTModel() {
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [testResults, setTestResults] = useState(null);
  const [error, setError] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, totalEpochs: 0 });
  
  const currentModelRef = useRef(null);

  /**
   * Create a new model
   */
  const createModel = useCallback((layerConfig = null) => {
    try {
      let newModel;
      if (layerConfig && layerConfig.length > 0) {
        // Create dynamic model from layer configuration
        newModel = createDynamicModel(layerConfig);
      } else {
        // Create default model
        newModel = createMNISTModel();
      }
      
      setModel(newModel);
      currentModelRef.current = newModel;
      setError(null);
      
      // Clear previous results
      setTrainingHistory([]);
      setTestResults(null);
      
      return newModel;
    } catch (err) {
      setError(`Failed to create model: ${err.message}`);
      throw err;
    }
  }, []);

  /**
   * Train the model
   */
  const trainModel = useCallback(async (trainData) => {
    if (!trainData) {
      throw new Error('Training data is required');
    }

    if (isTraining) {
      return;
    }

    setIsTraining(true);
    setError(null);
    setTrainingHistory([]);
    setTrainingProgress({ epoch: 0, totalEpochs: 10 });

    try {
      // Create new model if none exists
      const modelToTrain = model || createModel();
      
      const onEpochEnd = (epoch, logs) => {
        const logEntry = {
          epoch: epoch + 1,
          loss: logs.loss,
          accuracy: logs.acc,
          valLoss: logs.val_loss,
          valAccuracy: logs.val_acc,
          timestamp: new Date().toISOString()
        };
        
        setTrainingHistory(prev => [...prev, logEntry]);
        setTrainingProgress({ epoch: epoch + 1, totalEpochs: 10 });
      };

      await trainMNISTModel(modelToTrain, trainData, onEpochEnd);
      
      setTrainingProgress({ epoch: 10, totalEpochs: 10 });
      
    } catch (err) {
      setError(`Training failed: ${err.message}`);
      console.error('Training error:', err);
    } finally {
      setIsTraining(false);
    }
  }, [model, isTraining, createModel]);

  /**
   * Test the model
   */
  const testModel = useCallback(async (testData) => {
    if (!model || !testData) {
      throw new Error('Model and test data are required');
    }

    if (isTesting) {
      return;
    }

    setIsTesting(true);
    setError(null);

    try {
      const results = await evaluateModel(model, testData);
      setTestResults({
        ...results,
        timestamp: new Date().toISOString()
      });
      
      return results;
    } catch (err) {
      setError(`Testing failed: ${err.message}`);
      console.error('Testing error:', err);
      throw err;
    } finally {
      setIsTesting(false);
    }
  }, [model, isTesting]);

  /**
   * Make prediction on single image
   */
  const predict = useCallback(async (imageData) => {
    if (!model) {
      throw new Error('Model not available for prediction');
    }

    try {
      return await predictSingle(model, imageData);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      throw err;
    }
  }, [model]);

  /**
   * Save model to local storage or download
   */
  const saveModel = useCallback(async (name = 'mnist-model') => {
    if (!model) {
      throw new Error('No model to save');
    }

    try {
      await model.save(`localstorage://${name}`);
      return `Model saved as ${name}`;
    } catch (err) {
      setError(`Failed to save model: ${err.message}`);
      throw err;
    }
  }, [model]);

  /**
   * Load model from local storage
   */
  const loadModel = useCallback(async (name = 'mnist-model') => {
    try {
      const loadedModel = await tf.loadLayersModel(`localstorage://${name}`);
      setModel(loadedModel);
      currentModelRef.current = loadedModel;
      setError(null);
      
      return loadedModel;
    } catch (err) {
      setError(`Failed to load model: ${err.message}`);
      throw err;
    }
  }, []);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * Reset model state
   */
  const resetModel = useCallback(() => {
    if (currentModelRef.current) {
      currentModelRef.current.dispose();
    }
    
    setModel(null);
    currentModelRef.current = null;
    setTrainingHistory([]);
    setTestResults(null);
    setError(null);
    setTrainingProgress({ epoch: 0, totalEpochs: 0 });
  }, []);

  // Model statistics
  const modelStats = {
    isReady: !!model,
    isTraining,
    isTesting,
    hasHistory: trainingHistory.length > 0,
    hasTestResults: !!testResults,
    lastTrainingAccuracy: trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].accuracy : null,
    testAccuracy: testResults?.accuracy || null
  };

  return {
    // State
    model,
    isTraining,
    isTesting,
    trainingHistory,
    testResults,
    error,
    trainingProgress,
    modelStats,
    
    // Methods
    createModel,
    trainModel,
    testModel,
    predict,
    saveModel,
    loadModel,
    clearError,
    resetModel
  };
}