import { useState, useEffect, useCallback, useRef } from 'react';
import { MNISTDataLoader } from '../utils/mnistDataLoader.js';

/**
 * Custom hook for managing MNIST data loading and state
 * @returns {Object} Data loading state and methods
 */
export function useMNISTData() {
  const [trainData, setTrainData] = useState(null);
  const [testData, setTestData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  
  const dataLoaderRef = useRef(null);

  /**
   * Load MNIST data
   */
  const loadData = useCallback(async () => {
    if (isLoading || trainData || testData) return; // Prevent loading if already loaded
    
    setIsLoading(true);
    setError(null);
    setProgress(0);

    try {
      // Create new data loader instance
      dataLoaderRef.current = new MNISTDataLoader();
      
      // Simulate progress updates during loading
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const { trainData: newTrainData, testData: newTestData } = await dataLoaderRef.current.loadData();
      
      clearInterval(progressInterval);
      setProgress(100);
      
      setTrainData(newTrainData);
      setTestData(newTestData);
      
      // Reset progress after a brief delay
      setTimeout(() => setProgress(0), 1000);
      
    } catch (err) {
      setError(err.message);
      console.error('Failed to load MNIST data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, trainData, testData]);

  /**
   * Get a random test sample for visualization
   */
  const getRandomTestSample = useCallback(async () => {
    if (!dataLoaderRef.current || !testData) {
      throw new Error('Data not loaded');
    }
    
    try {
      return await dataLoaderRef.current.getRandomTestSample();
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, [testData]);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * Reset data state
   */
  const resetData = useCallback(() => {
    if (dataLoaderRef.current) {
      dataLoaderRef.current.dispose();
      dataLoaderRef.current = null;
    }
    setTrainData(null);
    setTestData(null);
    setError(null);
    setProgress(0);
  }, []);

  // Auto-load data on mount
  useEffect(() => {
    loadData();
    
    // Cleanup on unmount
    return () => {
      if (dataLoaderRef.current) {
        dataLoaderRef.current.dispose();
      }
    };
  }, []); // Empty dependency array to run only once

  // Provide data statistics
  const dataStats = {
    isLoaded: !!(trainData && testData),
    trainSize: trainData?.images.shape[0] || 0,
    testSize: testData?.images.shape[0] || 0,
    imageSize: trainData?.images.shape[1] || 0,
    numClasses: trainData?.labels.shape[1] || 0
  };

  return {
    // Data
    trainData,
    testData,
    
    // State
    isLoading,
    error,
    progress,
    dataStats,
    
    // Methods
    loadData,
    getRandomTestSample,
    clearError,
    resetData
  };
}