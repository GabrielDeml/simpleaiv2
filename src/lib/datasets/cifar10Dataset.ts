/**
 * CIFAR-10 dataset implementation using the standardized Dataset interface.
 * Wraps the existing Cifar10Data loader for compatibility.
 */

import { Dataset, registerDataset } from './datasetInterface';
import type { DatasetArrays, DatasetLoadOptions } from './datasetInterface';
import { Cifar10Data } from './cifar10';

/**
 * CIFAR-10 dataset class implementing the standard Dataset interface.
 * The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes,
 * with 6,000 images per class.
 */
export class CIFAR10Dataset extends Dataset {
  private cifar10Data: Cifar10Data | null = null;
  
  constructor() {
    super({
      name: 'CIFAR-10',
      description: '32x32 color images in 10 classes of common objects',
      inputShape: [32, 32],
      channels: 3,
      numClasses: 10,
      trainSize: 50000,
      testSize: 10000,
      classNames: [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
      ]
    });
  }
  
  /**
   * Load the CIFAR-10 dataset.
   * Uses the existing Cifar10Data loader and converts to standard format.
   */
  async loadData(options?: DatasetLoadOptions): Promise<DatasetArrays> {
    // Check cache first
    if (this.cachedData && options?.cache !== false) {
      return this.cachedData;
    }
    
    // Load using existing CIFAR-10 loader
    if (!this.cifar10Data) {
      this.cifar10Data = new Cifar10Data();
      await this.cifar10Data.load();
    }
    
    // Get raw arrays from the loader
    const trainData = this.cifar10Data.getTrainData();
    const testData = this.cifar10Data.getTestData();
    
    // Convert tensors to arrays
    const trainImages = await trainData.xs.data() as Float32Array;
    const trainLabels = await trainData.ys.data() as Float32Array;
    const testImages = await testData.xs.data() as Float32Array;
    const testLabels = await testData.ys.data() as Float32Array;
    
    // Dispose tensors to free memory
    trainData.xs.dispose();
    trainData.ys.dispose();
    testData.xs.dispose();
    testData.ys.dispose();
    
    // Apply sampling if requested
    let finalTrainImages = trainImages;
    let finalTrainLabels = trainLabels;
    let finalTestImages = testImages;
    let finalTestLabels = testLabels;
    
    if (options?.trainSampleRatio && options.trainSampleRatio < 1) {
      const trainSamples = Math.floor(this.metadata.trainSize * options.trainSampleRatio);
      const imageSize = 32 * 32 * 3;
      finalTrainImages = trainImages.slice(0, trainSamples * imageSize);
      finalTrainLabels = trainLabels.slice(0, trainSamples * this.metadata.numClasses);
    }
    
    if (options?.testSampleRatio && options.testSampleRatio < 1) {
      const testSamples = Math.floor(this.metadata.testSize * options.testSampleRatio);
      const imageSize = 32 * 32 * 3;
      finalTestImages = testImages.slice(0, testSamples * imageSize);
      finalTestLabels = testLabels.slice(0, testSamples * this.metadata.numClasses);
    }
    
    // Apply shuffling if requested
    let result: DatasetArrays = {
      trainImages: finalTrainImages,
      trainLabels: finalTrainLabels,
      testImages: finalTestImages,
      testLabels: finalTestLabels
    };
    
    if (options?.shuffle !== false) {
      const shuffledTrain = this.shuffleData(result.trainImages, result.trainLabels, options?.seed);
      const shuffledTest = this.shuffleData(result.testImages, result.testLabels, options?.seed);
      
      result = {
        trainImages: shuffledTrain.images,
        trainLabels: shuffledTrain.labels,
        testImages: shuffledTest.images,
        testLabels: shuffledTest.labels
      };
    }
    
    // Cache if requested
    if (options?.cache !== false) {
      this.cachedData = result;
    }
    
    return result;
  }
  
  /**
   * Clear cache and dispose of resources.
   */
  clearCache(): void {
    super.clearCache();
    this.cifar10Data = null;
  }
}

// Register the dataset
registerDataset('cifar10', () => new CIFAR10Dataset());