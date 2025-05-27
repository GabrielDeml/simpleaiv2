/**
 * Dataset interface for standardizing data loading across different datasets.
 * All dataset loaders must implement this interface to ensure consistency.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Represents the raw data structure returned by dataset loaders.
 * Arrays are flattened and need to be reshaped into tensors.
 */
export interface DatasetArrays {
  /** Flattened array of training images (normalized 0-1) */
  trainImages: Float32Array;
  /** Flattened array of training labels (one-hot encoded) */
  trainLabels: Float32Array;
  /** Flattened array of test images (normalized 0-1) */
  testImages: Float32Array;
  /** Flattened array of test labels (one-hot encoded) */
  testLabels: Float32Array;
}

/**
 * Metadata about the dataset structure and properties.
 */
export interface DatasetMetadata {
  /** Name of the dataset for display */
  name: string;
  /** Description of the dataset */
  description: string;
  /** Shape of a single input sample (e.g., [28, 28] for MNIST) */
  inputShape: number[];
  /** Number of channels (1 for grayscale, 3 for RGB) */
  channels: number;
  /** Number of output classes */
  numClasses: number;
  /** Number of training samples */
  trainSize: number;
  /** Number of test samples */
  testSize: number;
  /** Human-readable class names */
  classNames: string[];
}

/**
 * Structured tensor data ready for model training.
 */
export interface DatasetTensors {
  /** Training data tensor - 4D for images [batch, height, width, channels] or 2D for text [batch, sequence_length] */
  trainData: tf.Tensor4D | tf.Tensor2D;
  /** 2D tensor of training labels [batch, numClasses] */
  trainLabels: tf.Tensor2D;
  /** Test data tensor - 4D for images [batch, height, width, channels] or 2D for text [batch, sequence_length] */
  testData: tf.Tensor4D | tf.Tensor2D;
  /** 2D tensor of test labels [batch, numClasses] */
  testLabels: tf.Tensor2D;
}

/**
 * Configuration options for dataset loading.
 */
export interface DatasetLoadOptions {
  /** Whether to shuffle the data (default: true) */
  shuffle?: boolean;
  /** Optional seed for reproducible shuffling */
  seed?: number;
  /** Percentage of training data to use (0-1, default: 1) */
  trainSampleRatio?: number;
  /** Percentage of test data to use (0-1, default: 1) */
  testSampleRatio?: number;
  /** Whether to cache the loaded data (default: true) */
  cache?: boolean;
}

/**
 * Abstract base class for dataset loaders.
 * Provides common functionality and enforces consistent interface.
 */
export abstract class Dataset {
  protected metadata: DatasetMetadata;
  protected cachedData: DatasetArrays | null = null;
  
  constructor(metadata: DatasetMetadata) {
    this.metadata = metadata;
  }
  
  /**
   * Get dataset metadata.
   */
  getMetadata(): DatasetMetadata {
    return this.metadata;
  }
  
  /**
   * Load the raw dataset arrays.
   * Must be implemented by each dataset loader.
   */
  abstract loadData(options?: DatasetLoadOptions): Promise<DatasetArrays>;
  
  /**
   * Convert raw arrays to TensorFlow tensors.
   * Handles reshaping and normalization.
   * Uses tf.tidy() to prevent memory leaks from intermediate tensors.
   */
  async loadTensors(options?: DatasetLoadOptions): Promise<DatasetTensors> {
    const data = await this.loadData(options);
    
    const { inputShape, channels } = this.metadata;
    const imageSize = inputShape.reduce((a, b) => a * b, 1) * channels;
    
    // Calculate actual sizes from data
    const trainSamples = data.trainImages.length / imageSize;
    const testSamples = data.testImages.length / imageSize;
    
    // Use tf.tidy to ensure no intermediate tensors leak
    // Note: We need to keep the returned tensors, so we create them outside tidy
    let trainData: tf.Tensor4D;
    let trainLabels: tf.Tensor2D;
    let testData: tf.Tensor4D;
    let testLabels: tf.Tensor2D;
    
    tf.tidy(() => {
      // Create tensors inside tidy to catch any intermediate operations
      trainData = tf.tensor4d(
        data.trainImages,
        [trainSamples, ...inputShape, channels] as [number, number, number, number]
      );
      
      trainLabels = tf.tensor2d(
        data.trainLabels,
        [trainSamples, this.metadata.numClasses]
      );
      
      testData = tf.tensor4d(
        data.testImages,
        [testSamples, ...inputShape, channels] as [number, number, number, number]
      );
      
      testLabels = tf.tensor2d(
        data.testLabels,
        [testSamples, this.metadata.numClasses]
      );
      
      // Keep tensors from being disposed by tidy
      tf.keep(trainData);
      tf.keep(trainLabels);
      tf.keep(testData);
      tf.keep(testLabels);
    });
    
    return { 
      trainData: trainData!, 
      trainLabels: trainLabels!, 
      testData: testData!, 
      testLabels: testLabels! 
    };
  }
  
  /**
   * Utility method to one-hot encode labels.
   */
  protected oneHotEncode(labels: Uint8Array | number[], numClasses: number): Float32Array {
    const encoded = new Float32Array(labels.length * numClasses);
    for (let i = 0; i < labels.length; i++) {
      encoded[i * numClasses + labels[i]] = 1;
    }
    return encoded;
  }
  
  /**
   * Utility method to shuffle data and labels together.
   */
  protected shuffleData(
    images: Float32Array,
    labels: Float32Array,
    seed?: number
  ): { images: Float32Array; labels: Float32Array } {
    const numSamples = labels.length / this.metadata.numClasses;
    const imageSize = images.length / numSamples;
    
    // Create indices array
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    
    // Shuffle indices using Fisher-Yates algorithm
    if (seed !== undefined) {
      // Simple seeded random number generator for reproducible shuffling
      let seedValue = seed;
      const seededRandom = () => {
        seedValue = (seedValue * 9301 + 49297) % 233280;
        return seedValue / 233280;
      };
      
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    } else {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }
    
    // Create shuffled arrays
    const shuffledImages = new Float32Array(images.length);
    const shuffledLabels = new Float32Array(labels.length);
    
    for (let i = 0; i < numSamples; i++) {
      const srcIdx = indices[i];
      
      // Copy image data
      const imgSrcStart = srcIdx * imageSize;
      const imgDstStart = i * imageSize;
      for (let j = 0; j < imageSize; j++) {
        shuffledImages[imgDstStart + j] = images[imgSrcStart + j];
      }
      
      // Copy label data
      const lblSrcStart = srcIdx * this.metadata.numClasses;
      const lblDstStart = i * this.metadata.numClasses;
      for (let j = 0; j < this.metadata.numClasses; j++) {
        shuffledLabels[lblDstStart + j] = labels[lblSrcStart + j];
      }
    }
    
    return { images: shuffledImages, labels: shuffledLabels };
  }
  
  /**
   * Clear cached data to free memory.
   */
  clearCache(): void {
    this.cachedData = null;
  }
}

/**
 * Factory function type for creating dataset instances.
 */
export type DatasetFactory = () => Dataset;

/**
 * Registry of available datasets.
 */
export const datasetRegistry: Map<string, DatasetFactory> = new Map();

/**
 * Register a dataset factory.
 */
export function registerDataset(name: string, factory: DatasetFactory): void {
  datasetRegistry.set(name, factory);
}

/**
 * Get a dataset instance by name.
 */
export function getDataset(name: string): Dataset {
  const factory = datasetRegistry.get(name);
  if (!factory) {
    throw new Error(`Unknown dataset: ${name}. Available: ${Array.from(datasetRegistry.keys()).join(', ')}`);
  }
  return factory();
}