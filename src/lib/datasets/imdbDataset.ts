import * as tf from '@tensorflow/tfjs';
import { Dataset, registerDataset } from './datasetInterface';
import type { DatasetArrays, DatasetTensors, DatasetLoadOptions } from './datasetInterface';

/**
 * Simple tokenizer for text data
 * Converts text to sequences of integer indices
 */
export class SimpleTokenizer {
  private wordIndex: Map<string, number> = new Map();
  private indexWord: Map<number, string> = new Map();
  private vocabSize: number;
  private oovToken = '<OOV>';
  private padToken = '<PAD>';
  
  constructor(vocabSize: number = 10000) {
    this.vocabSize = vocabSize;
    // Reserve indices for special tokens
    this.wordIndex.set(this.padToken, 0);
    this.wordIndex.set(this.oovToken, 1);
    this.indexWord.set(0, this.padToken);
    this.indexWord.set(1, this.oovToken);
  }
  
  /**
   * Fit tokenizer on texts to build vocabulary
   */
  fitOnTexts(texts: string[]): void {
    const wordCounts = new Map<string, number>();
    
    // Count word frequencies
    for (const text of texts) {
      const words = this.preprocessText(text);
      for (const word of words) {
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      }
    }
    
    // Sort by frequency and take top vocabSize words
    const sortedWords = Array.from(wordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.vocabSize - 2); // Reserve 2 for special tokens
    
    // Build word index
    let index = 2; // Start after special tokens
    for (const [word] of sortedWords) {
      this.wordIndex.set(word, index);
      this.indexWord.set(index, word);
      index++;
    }
  }
  
  /**
   * Convert texts to sequences of integers
   */
  textsToSequences(texts: string[]): number[][] {
    return texts.map(text => {
      const words = this.preprocessText(text);
      return words.map(word => this.wordIndex.get(word) || 1); // Use OOV token for unknown words
    });
  }
  
  /**
   * Pad sequences to same length
   */
  padSequences(sequences: number[][], maxLength: number): number[][] {
    return sequences.map(seq => {
      if (seq.length > maxLength) {
        return seq.slice(0, maxLength);
      } else if (seq.length < maxLength) {
        const padding = new Array(maxLength - seq.length).fill(0);
        return [...seq, ...padding];
      }
      return seq;
    });
  }
  
  private preprocessText(text: string): string[] {
    // Simple preprocessing: lowercase and split by whitespace
    return text.toLowerCase()
      .replace(/[^\w\s]/g, '') // Remove punctuation
      .split(/\s+/)
      .filter(word => word.length > 0);
  }
}

/**
 * IMDB Movie Review Dataset for sentiment analysis
 * Binary classification: positive (1) or negative (0) reviews
 */
export class IMDBDataset extends Dataset {
  private tokenizer: SimpleTokenizer;
  private maxLength: number = 200;
  
  constructor() {
    super({
      name: 'IMDB Movie Reviews',
      description: 'Binary sentiment classification of movie reviews',
      inputShape: [200], // Sequence length
      channels: 1,
      numClasses: 2,
      trainSize: 20000,
      testSize: 5000,
      classNames: ['Negative', 'Positive']
    });
    
    this.tokenizer = new SimpleTokenizer(10000);
  }
  
  /**
   * Generate synthetic IMDB-like data for demonstration
   * In production, load real IMDB dataset
   */
  async loadData(options?: DatasetLoadOptions): Promise<DatasetArrays> {
    // Check cache
    if (this.cachedData && options?.cache !== false) {
      return this.cachedData;
    }
    
    // Generate synthetic reviews for demonstration
    const trainTexts = this.generateSyntheticReviews(this.metadata.trainSize);
    const testTexts = this.generateSyntheticReviews(this.metadata.testSize);
    
    // Fit tokenizer on training data
    this.tokenizer.fitOnTexts(trainTexts.texts);
    
    // Convert to sequences
    const trainSequences = this.tokenizer.textsToSequences(trainTexts.texts);
    const testSequences = this.tokenizer.textsToSequences(testTexts.texts);
    
    // Pad sequences
    const paddedTrain = this.tokenizer.padSequences(trainSequences, this.maxLength);
    const paddedTest = this.tokenizer.padSequences(testSequences, this.maxLength);
    
    // Convert to Float32Array
    const trainImages = new Float32Array(paddedTrain.flat());
    const testImages = new Float32Array(paddedTest.flat());
    
    // One-hot encode labels
    const trainLabels = this.oneHotEncode(trainTexts.labels, this.metadata.numClasses);
    const testLabels = this.oneHotEncode(testTexts.labels, this.metadata.numClasses);
    
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
   * Generate synthetic movie reviews for demonstration
   */
  private generateSyntheticReviews(count: number): { texts: string[], labels: number[] } {
    const positiveWords = [
      'amazing', 'excellent', 'fantastic', 'wonderful', 'great', 'love', 'perfect',
      'brilliant', 'outstanding', 'superb', 'masterpiece', 'enjoyable', 'recommend'
    ];
    
    const negativeWords = [
      'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'boring',
      'disappointing', 'waste', 'poor', 'ridiculous', 'avoid', 'dull'
    ];
    
    const neutralWords = [
      'movie', 'film', 'acting', 'story', 'plot', 'characters', 'scenes',
      'director', 'production', 'performance', 'cinematography', 'script'
    ];
    
    const texts: string[] = [];
    const labels: number[] = [];
    
    for (let i = 0; i < count; i++) {
      const isPositive = i % 2 === 0;
      const sentimentWords = isPositive ? positiveWords : negativeWords;
      
      // Generate a review with 20-50 words
      const reviewLength = Math.floor(Math.random() * 30) + 20;
      const words: string[] = [];
      
      for (let j = 0; j < reviewLength; j++) {
        if (Math.random() < 0.3) {
          // Add sentiment word
          words.push(sentimentWords[Math.floor(Math.random() * sentimentWords.length)]);
        } else {
          // Add neutral word
          words.push(neutralWords[Math.floor(Math.random() * neutralWords.length)]);
        }
      }
      
      texts.push(words.join(' '));
      labels.push(isPositive ? 1 : 0);
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
registerDataset('imdb', () => new IMDBDataset());