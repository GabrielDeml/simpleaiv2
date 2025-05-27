/**
 * Fashion-MNIST dataset implementation using the standardized Dataset interface.
 * Wraps the existing FashionMnistData loader for compatibility.
 */

import { Dataset, registerDataset } from './datasetInterface';
import type { DatasetArrays, DatasetLoadOptions } from './datasetInterface';
import { FashionMnistData } from './fashionMnist';

/**
 * Fashion-MNIST dataset class implementing the standard Dataset interface.
 * Fashion-MNIST is a dataset of Zalando's article images, intended as a
 * drop-in replacement for MNIST.
 */
export class FashionMNISTDataset extends Dataset {
  private fashionMnistData: FashionMnistData | null = null;
  
  constructor() {
    super({
      name: 'Fashion-MNIST',
      description: 'Grayscale images of fashion products from Zalando',
      inputShape: [28, 28],
      channels: 1,
      numClasses: 10,
      trainSize: 60000,
      testSize: 10000,
      classNames: [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
      ]
    });
  }
  
  /**
   * Load the Fashion-MNIST dataset.
   * Uses the existing FashionMnistData loader and converts to standard format.
   */
  async loadData(options?: DatasetLoadOptions): Promise<DatasetArrays> {
    // Check cache first
    if (this.cachedData && options?.cache !== false) {
      return this.cachedData;
    }
    
    // Load using existing Fashion-MNIST loader
    if (!this.fashionMnistData) {
      this.fashionMnistData = new FashionMnistData();
      await this.fashionMnistData.load();
    }
    
    // Get raw arrays from the loader
    const trainData = this.fashionMnistData.getTrainData();
    const testData = this.fashionMnistData.getTestData();
    
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
      const imageSize = 28 * 28;
      finalTrainImages = trainImages.slice(0, trainSamples * imageSize);
      finalTrainLabels = trainLabels.slice(0, trainSamples * this.metadata.numClasses);
    }
    
    if (options?.testSampleRatio && options.testSampleRatio < 1) {
      const testSamples = Math.floor(this.metadata.testSize * options.testSampleRatio);
      const imageSize = 28 * 28;
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
    this.fashionMnistData = null;
  }
}

// Register the dataset
registerDataset('fashion-mnist', () => new FashionMNISTDataset());