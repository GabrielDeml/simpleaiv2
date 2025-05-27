import * as tf from '@tensorflow/tfjs';
import { Dataset, registerDataset } from './datasetInterface';
import type { DatasetArrays, DatasetTensors, DatasetLoadOptions } from './datasetInterface';
import { SimpleTokenizer } from './imdbDataset';

/**
 * AG News Dataset for text classification
 * 4-class classification of news articles into categories
 */
export class AGNewsDataset extends Dataset {
  private tokenizer: SimpleTokenizer;
  private maxLength: number = 150;
  
  constructor() {
    super({
      name: 'AG News',
      description: '4-class news categorization dataset',
      inputShape: [150], // Sequence length
      channels: 1,
      numClasses: 4,
      trainSize: 20000,
      testSize: 5000,
      classNames: ['World', 'Sports', 'Business', 'Science/Tech']
    });
    
    this.tokenizer = new SimpleTokenizer(10000);
  }
  
  /**
   * Generate synthetic AG News-like data for demonstration
   * In production, load real AG News dataset
   */
  async loadData(options?: DatasetLoadOptions): Promise<DatasetArrays> {
    // Check cache
    if (this.cachedData && options?.cache !== false) {
      return this.cachedData;
    }
    
    // Generate synthetic news articles
    const trainData = this.generateSyntheticNews(this.metadata.trainSize);
    const testData = this.generateSyntheticNews(this.metadata.testSize);
    
    // Fit tokenizer on training data
    this.tokenizer.fitOnTexts(trainData.texts);
    
    // Convert to sequences
    const trainSequences = this.tokenizer.textsToSequences(trainData.texts);
    const testSequences = this.tokenizer.textsToSequences(testData.texts);
    
    // Pad sequences
    const paddedTrain = this.tokenizer.padSequences(trainSequences, this.maxLength);
    const paddedTest = this.tokenizer.padSequences(testSequences, this.maxLength);
    
    // Convert to Float32Array
    const trainImages = new Float32Array(paddedTrain.flat());
    const testImages = new Float32Array(paddedTest.flat());
    
    // One-hot encode labels
    const trainLabels = this.oneHotEncode(trainData.labels, this.metadata.numClasses);
    const testLabels = this.oneHotEncode(testData.labels, this.metadata.numClasses);
    
    const data: DatasetArrays = {
      trainImages,
      trainLabels,
      testImages,
      testLabels
    };
    
    // Apply options
    if (options?.shuffle) {
      const trainShuffled = this.shuffleData(trainImages, trainLabels, options.seed);
      data.trainImages = trainShuffled.images;
      data.trainLabels = trainShuffled.labels;
    }
    
    // Cache if requested
    if (options?.cache !== false) {
      this.cachedData = data;
    }
    
    return data;
  }
  
  /**
   * Generate synthetic news articles for demonstration
   */
  private generateSyntheticNews(count: number): { texts: string[], labels: number[] } {
    const categoryWords = {
      0: [ // World
        'country', 'government', 'president', 'minister', 'election', 'policy',
        'international', 'summit', 'treaty', 'diplomacy', 'conflict', 'peace'
      ],
      1: [ // Sports
        'game', 'player', 'team', 'score', 'championship', 'tournament',
        'match', 'victory', 'defeat', 'season', 'league', 'coach'
      ],
      2: [ // Business
        'company', 'market', 'stock', 'investment', 'profit', 'revenue',
        'CEO', 'merger', 'acquisition', 'economy', 'finance', 'growth'
      ],
      3: [ // Science/Tech
        'technology', 'research', 'scientist', 'study', 'discovery', 'innovation',
        'software', 'AI', 'data', 'experiment', 'breakthrough', 'development'
      ]
    };
    
    const commonWords = [
      'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'with',
      'on', 'as', 'by', 'at', 'from', 'has', 'was', 'are', 'been', 'will'
    ];
    
    const texts: string[] = [];
    const labels: number[] = [];
    
    for (let i = 0; i < count; i++) {
      const category = i % 4;
      const catWords = categoryWords[category as keyof typeof categoryWords];
      
      // Generate article with 30-60 words
      const articleLength = Math.floor(Math.random() * 30) + 30;
      const words: string[] = [];
      
      for (let j = 0; j < articleLength; j++) {
        if (Math.random() < 0.3) {
          // Add category-specific word
          words.push(catWords[Math.floor(Math.random() * catWords.length)]);
        } else {
          // Add common word
          words.push(commonWords[Math.floor(Math.random() * commonWords.length)]);
        }
      }
      
      texts.push(words.join(' '));
      labels.push(category);
    }
    
    return { texts, labels };
  }

  /**
   * Override loadTensors for text data (2D instead of 4D)
   */
  async loadTensors(options?: DatasetLoadOptions): Promise<DatasetTensors> {
    const data = await this.loadData(options);
    
    // Calculate actual sizes from data
    const trainSamples = data.trainImages.length / this.maxLength;
    const testSamples = data.testImages.length / this.maxLength;
    
    // Use tf.tidy to ensure no intermediate tensors leak
    let trainData!: tf.Tensor2D;
    let trainLabels!: tf.Tensor2D;
    let testData!: tf.Tensor2D;
    let testLabels!: tf.Tensor2D;
    
    tf.tidy(() => {
      // Create 2D tensors for text data [samples, sequence_length]
      trainData = tf.tensor2d(
        data.trainImages,
        [trainSamples, this.maxLength]
      );
      
      trainLabels = tf.tensor2d(
        data.trainLabels,
        [trainSamples, this.metadata.numClasses]
      );
      
      testData = tf.tensor2d(
        data.testImages,
        [testSamples, this.maxLength]
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
      trainData,
      trainLabels,
      testData,
      testLabels
    };
  }
}

// Register the dataset
registerDataset('ag-news', () => new AGNewsDataset());