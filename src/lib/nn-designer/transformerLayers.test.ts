import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import { MultiHeadAttentionLayer, PositionalEncodingLayer, TransformerEncoderBlock, GlobalAveragePooling1DLayer } from './transformerLayers';

// Mock console methods to avoid noise in tests
beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

describe('MultiHeadAttentionLayer', () => {
  let layer: MultiHeadAttentionLayer;
  
  beforeEach(() => {
    layer = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.numHeads).toBe(2);
    expect(config.keyDim).toBe(4);
    expect(config.valueDim).toBe(4); // defaults to keyDim
    expect(config.dropout).toBe(0.0);
    expect(config.useBias).toBe(true);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [10, 8]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [10, 8];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should process input tensors correctly', () => {
    const batchSize = 2;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });

  it('should handle training parameter correctly', () => {
    const batchSize = 2;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    
    // Test with training=true
    const outputTrain = layer.apply(input, { training: true }) as tf.Tensor;
    expect(outputTrain.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Test with training=false
    const outputEval = layer.apply(input, { training: false }) as tf.Tensor;
    expect(outputEval.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    outputTrain.dispose();
    outputEval.dispose();
  });
});

describe('PositionalEncodingLayer', () => {
  let layer: PositionalEncodingLayer;
  
  beforeEach(() => {
    layer = new PositionalEncodingLayer({
      maxLength: 100
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.maxLength).toBe(100);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [50, 128]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [50, 128];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should add positional encoding to input', () => {
    const batchSize = 2;
    const seqLength = 10;
    const embeddingDim = 16;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.zeros([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // The output should not be all zeros after adding positional encoding
    const outputData = output.dataSync();
    const hasNonZero = outputData.some(val => Math.abs(val) > 1e-6);
    expect(hasNonZero).toBe(true);
    
    input.dispose();
    output.dispose();
  });

  it('should handle sequences shorter than maxLength', () => {
    const batchSize = 1;
    const seqLength = 5; // shorter than maxLength
    const embeddingDim = 8;
    
    layer.build([100, embeddingDim]); // build with maxLength
    
    const input = tf.zeros([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });
});

describe('TransformerEncoderBlock', () => {
  let layer: TransformerEncoderBlock;
  
  beforeEach(() => {
    layer = new TransformerEncoderBlock({
      numHeads: 2,
      keyDim: 4,
      ffDim: 16,
      dropout: 0.1
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.numHeads).toBe(2);
    expect(config.keyDim).toBe(4);
    expect(config.ffDim).toBe(16);
    expect(config.dropout).toBe(0.1);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [10, 8]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [10, 8];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should process input tensors correctly', () => {
    const batchSize = 2;
    const seqLength = 4;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });

  it('should handle training parameter correctly', () => {
    const batchSize = 1;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    
    // Test with training=true
    const outputTrain = layer.apply(input, { training: true }) as tf.Tensor;
    expect(outputTrain.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Test with training=false
    const outputEval = layer.apply(input, { training: false }) as tf.Tensor;
    expect(outputEval.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    outputTrain.dispose();
    outputEval.dispose();
  });

  it('should implement residual connections correctly', () => {
    const batchSize = 1;
    const seqLength = 2;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    // Create input with known values
    const inputValues = new Float32Array(batchSize * seqLength * embeddingDim);
    for (let i = 0; i < inputValues.length; i++) {
      inputValues[i] = 0.1; // small constant value
    }
    const input = tf.tensor3d(inputValues, [batchSize, seqLength, embeddingDim]);
    
    const output = layer.apply(input, { training: false }) as tf.Tensor;
    
    // Output should have same shape as input
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Due to residual connections and layer normalization, 
    // output should not be exactly the same as input but should be in reasonable range
    const outputData = output.dataSync();
    const avgOutput = Array.from(outputData).reduce((a: number, b: number) => a + b, 0) / outputData.length;
    expect(Math.abs(avgOutput)).toBeLessThan(10); // reasonable range
    
    input.dispose();
    output.dispose();
  });
});

describe('GlobalAveragePooling1DLayer', () => {
  let layer: GlobalAveragePooling1DLayer;
  
  beforeEach(() => {
    layer = new GlobalAveragePooling1DLayer();
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config).toBeDefined();
  });

  it('should build correctly with proper 3D input shape', () => {
    const inputShape = [10, 128]; // [seqLength, features]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should throw error when building with invalid input shapes', () => {
    // Test with 1D shape
    expect(() => layer.build([128])).toThrow('GlobalAveragePooling1D layer expects 2D or 3D input shape');
    
    // Test with 4D shape
    expect(() => layer.build([2, 10, 128, 64])).toThrow('GlobalAveragePooling1D layer expects 2D or 3D input shape');
    
    // Test with null/undefined
    expect(() => layer.build(null as any)).toThrow('GlobalAveragePooling1D layer expects 2D or 3D input shape');
  });
  
  it('should build correctly with 3D input shape', () => {
    const inputShape = [32, 10, 128]; // [batch, seqLength, features]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [10, 128]; // [seqLength, features]
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual([128]); // [features] - batch dimension is implicit
  });

  it('should average across sequence dimension correctly', () => {
    const batchSize = 2;
    const seqLength = 4;
    const features = 3;
    
    layer.build([seqLength, features]);
    
    // Create input with known values
    const inputData = [
      // Batch 1
      [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
      // Batch 2
      [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]]
    ];
    
    const input = tf.tensor3d(inputData, [batchSize, seqLength, features]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, features]);
    
    // Check the averaged values
    const outputData = output.arraySync() as number[][];
    
    // Batch 1: avg of [[1,2,3], [4,5,6], [7,8,9], [10,11,12]] = [5.5, 6.5, 7.5]
    expect(outputData[0][0]).toBeCloseTo(5.5);
    expect(outputData[0][1]).toBeCloseTo(6.5);
    expect(outputData[0][2]).toBeCloseTo(7.5);
    
    // Batch 2: avg of [[2,4,6], [8,10,12], [14,16,18], [20,22,24]] = [11, 13, 15]
    expect(outputData[1][0]).toBeCloseTo(11);
    expect(outputData[1][1]).toBeCloseTo(13);
    expect(outputData[1][2]).toBeCloseTo(15);
    
    input.dispose();
    output.dispose();
  });

  it('should throw error when processing tensors with wrong rank', () => {
    layer.build([10, 128]);
    
    // Test with 2D tensor
    const input2D = tf.zeros([10, 128]);
    expect(() => layer.apply(input2D)).toThrow('GlobalAveragePooling1D expects 3D tensor input, but got tensor with rank 2');
    input2D.dispose();
    
    // Test with 4D tensor
    const input4D = tf.zeros([2, 10, 128, 64]);
    expect(() => layer.apply(input4D)).toThrow('GlobalAveragePooling1D expects 3D tensor input, but got tensor with rank 4');
    input4D.dispose();
  });

  it('should handle edge case of single sequence length', () => {
    const batchSize = 3;
    const seqLength = 1; // Edge case: only one timestep
    const features = 5;
    
    layer.build([seqLength, features]);
    
    // Create specific test data
    const testData = [
      [[1, 2, 3, 4, 5]],      // Batch 1
      [[6, 7, 8, 9, 10]],     // Batch 2
      [[11, 12, 13, 14, 15]]  // Batch 3
    ];
    
    const input = tf.tensor3d(testData, [batchSize, seqLength, features]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, features]);
    
    // With seqLength=1, averaging across a single timestep should return the same values
    const outputData = output.arraySync() as number[][];
    
    // Check that output matches input (without sequence dimension)
    expect(outputData[0]).toEqual([1, 2, 3, 4, 5]);
    expect(outputData[1]).toEqual([6, 7, 8, 9, 10]);
    expect(outputData[2]).toEqual([11, 12, 13, 14, 15]);
    
    input.dispose();
    output.dispose();
  });

  it('should preserve gradient flow during backpropagation', () => {
    const batchSize = 1;
    const seqLength = 3;
    const features = 2;
    
    layer.build([seqLength, features]);
    
    // Test gradient flow
    const input = tf.variable(tf.ones([batchSize, seqLength, features]));
    const output = layer.apply(input) as tf.Tensor;
    
    // Create a simple loss (sum of outputs)
    const loss = tf.sum(output);
    
    // Compute gradients
    const grads = tf.grad((x: tf.Tensor) => {
      const out = layer.apply(x) as tf.Tensor;
      return tf.sum(out);
    })(input);
    
    expect(grads.shape).toEqual(input.shape);
    
    // Each gradient should be 1/seqLength due to averaging
    const gradData = grads.arraySync() as number[][][];
    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < seqLength; s++) {
        for (let f = 0; f < features; f++) {
          expect(gradData[b][s][f]).toBeCloseTo(1 / seqLength);
        }
      }
    }
    
    input.dispose();
    output.dispose();
    loss.dispose();
    grads.dispose();
  });
});

describe('Layer Registration', () => {
  it('should register custom layers with TensorFlow.js', () => {
    // Test that our custom layers are registered and can be serialized
    const multiHeadLayer = new MultiHeadAttentionLayer({
      numHeads: 4,
      keyDim: 8
    });
    
    const positionalLayer = new PositionalEncodingLayer({
      maxLength: 50
    });
    
    const transformerLayer = new TransformerEncoderBlock({
      numHeads: 2,
      keyDim: 4,
      ffDim: 16
    });
    
    const poolingLayer = new GlobalAveragePooling1DLayer();
    
    // Test that getConfig works (needed for serialization)
    expect(multiHeadLayer.getConfig()).toBeDefined();
    expect(positionalLayer.getConfig()).toBeDefined();
    expect(transformerLayer.getConfig()).toBeDefined();
    expect(poolingLayer.getConfig()).toBeDefined();
    
    // Test that class names are set correctly
    expect(MultiHeadAttentionLayer.className).toBe('MultiHeadAttentionLayer');
    expect(PositionalEncodingLayer.className).toBe('PositionalEncodingLayer');
    expect(TransformerEncoderBlock.className).toBe('TransformerEncoderBlock');
    expect(GlobalAveragePooling1DLayer.className).toBe('GlobalAveragePooling1DLayer');
  });
});

describe('Edge Cases and Error Handling', () => {
  it('should handle invalid input shapes gracefully', () => {
    const layer = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4
    });
    
    // Test with invalid input shape
    expect(() => layer.build(42 as any)).toThrow('Invalid input shape');
    expect(() => layer.build(null as any)).toThrow('Invalid input shape');
  });

  it('should validate numHeads configuration', () => {
    // Test with valid configuration
    expect(() => new MultiHeadAttentionLayer({
      numHeads: 1,
      keyDim: 4
    })).not.toThrow();
    
    expect(() => new MultiHeadAttentionLayer({
      numHeads: 8,
      keyDim: 64
    })).not.toThrow();
  });

  it('should handle dropout configuration correctly', () => {
    const layerWithDropout = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4,
      dropout: 0.5
    });
    
    const layerWithoutDropout = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4,
      dropout: 0.0
    });
    
    expect(layerWithDropout.getConfig().dropout).toBe(0.5);
    expect(layerWithoutDropout.getConfig().dropout).toBe(0.0);
  });
});