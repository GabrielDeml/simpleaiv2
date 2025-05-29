import * as tf from '@tensorflow/tfjs';

// Create base layer class if it doesn't exist
const LayerClass = tf.layers?.Layer || class Layer {
  constructor(config?: any) {}
  build(inputShape: tf.Shape | tf.Shape[]): void {}
  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor | tf.Tensor[] {
    throw new Error('Not implemented');
  }
  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }
  getConfig(): any {
    return {};
  }
  apply(inputs: tf.Tensor | tf.Tensor[], config?: any): tf.Tensor | tf.Tensor[] {
    return this.call(inputs, config);
  }
};

/**
 * Custom implementation of Multi-Head Attention for TensorFlow.js
 * Since tf.layers doesn't have built-in MultiHeadAttention, we implement it
 * using available operations like dense layers and matrix multiplication.
 */
export class MultiHeadAttentionLayer extends LayerClass {
  private numHeads: number;
  private keyDim: number;
  private valueDim: number;
  private dropout: number;
  private useBias: boolean;
  
  private queryDense!: tf.layers.Layer;
  private keyDense!: tf.layers.Layer;
  private valueDense!: tf.layers.Layer;
  private outputDense!: tf.layers.Layer;
  private dropoutLayer?: tf.layers.Layer;
  
  constructor(config: {
    numHeads: number;
    keyDim: number;
    valueDim?: number;
    dropout?: number;
    useBias?: boolean;
  }) {
    super({});
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.valueDim = config.valueDim || config.keyDim;
    this.dropout = config.dropout || 0.0;
    this.useBias = config.useBias !== false;
  }
  
  build(inputShape: tf.Shape | tf.Shape[]): void {
    // Handle both single shape and array of shapes
    let shape: tf.Shape;
    if (Array.isArray(inputShape)) {
      // If it's an array of shapes, use the first one
      // If it's just a simple shape array, use it directly
      if (Array.isArray(inputShape[0])) {
        shape = inputShape[0] as tf.Shape;
      } else {
        shape = inputShape as tf.Shape;
      }
    } else {
      shape = inputShape;
    }
    
    if (!shape || typeof shape === 'number' || shape.length < 2) {
      throw new Error('Invalid input shape');
    }
    const inputDim = shape[shape.length - 1] as number;
    
    // Create dense layers for Q, K, V projections
    this.queryDense = tf.layers.dense({
      units: this.numHeads * this.keyDim,
      useBias: this.useBias,
      name: 'query'
    });
    
    this.keyDense = tf.layers.dense({
      units: this.numHeads * this.keyDim,
      useBias: this.useBias,
      name: 'key'
    });
    
    this.valueDense = tf.layers.dense({
      units: this.numHeads * this.valueDim,
      useBias: this.useBias,
      name: 'value'
    });
    
    this.outputDense = tf.layers.dense({
      units: inputDim,
      useBias: this.useBias,
      name: 'output'
    });
    
    if (this.dropout > 0) {
      this.dropoutLayer = tf.layers.dropout({ rate: this.dropout });
    }
    
    // Build sub-layers
    this.queryDense.build(shape);
    this.keyDense.build(shape);
    this.valueDense.build(shape);
    const outputShape = [...shape.slice(0, -1), this.numHeads * this.valueDim] as tf.Shape;
    this.outputDense.build(outputShape);
    
    super.build(inputShape);
  }
  
  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const training = kwargs?.training || false;
      
      const batchSize = input.shape[0]!;
      const seqLength = input.shape[1]!;
      
      // Project inputs to Q, K, V
      const query = this.queryDense.apply(input) as tf.Tensor;
      const key = this.keyDense.apply(input) as tf.Tensor;
      const value = this.valueDense.apply(input) as tf.Tensor;
      
      // Reshape to separate heads
      const queryHeads = tf.reshape(query, [batchSize, seqLength, this.numHeads, this.keyDim]);
      const keyHeads = tf.reshape(key, [batchSize, seqLength, this.numHeads, this.keyDim]);
      const valueHeads = tf.reshape(value, [batchSize, seqLength, this.numHeads, this.valueDim]);
      
      // Transpose to [batch, heads, seq, dim]
      const queryT = tf.transpose(queryHeads, [0, 2, 1, 3]);
      const keyT = tf.transpose(keyHeads, [0, 2, 1, 3]);
      const valueT = tf.transpose(valueHeads, [0, 2, 1, 3]);
      
      // Compute attention scores
      const scores = tf.matMul(queryT, keyT, false, true);
      const scaledScores = tf.div(scores, tf.sqrt(tf.scalar(this.keyDim)));
      
      // Apply softmax
      const attentionWeights = tf.softmax(scaledScores, -1);
      
      // Apply dropout if training
      let weights = attentionWeights;
      if (this.dropout > 0 && this.dropoutLayer && training) {
        weights = this.dropoutLayer.apply(weights, { training }) as tf.Tensor;
      }
      
      // Apply attention to values
      const attentionOutput = tf.matMul(weights, valueT);
      
      // Transpose back and reshape
      const outputTransposed = tf.transpose(attentionOutput, [0, 2, 1, 3]);
      const concatenated = tf.reshape(outputTransposed, [batchSize, seqLength, this.numHeads * this.valueDim]);
      
      // Final projection
      return this.outputDense.apply(concatenated) as tf.Tensor;
    });
  }
  
  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }
  
  getConfig(): any {
    const config = super.getConfig();
    return {
      ...config,
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      dropout: this.dropout,
      useBias: this.useBias
    };
  }
  
  static className = 'MultiHeadAttentionLayer';
}

/**
 * Positional Encoding layer for transformer models
 * Adds sinusoidal position encodings to embeddings
 */
export class PositionalEncodingLayer extends LayerClass {
  private maxLength: number;
  private encodingMatrix: tf.Tensor2D | null = null;
  
  constructor(config: { maxLength: number }) {
    super({});
    this.maxLength = config.maxLength;
  }
  
  build(inputShape: tf.Shape | tf.Shape[]): void {
    // Handle both single shape and array of shapes
    let shape: tf.Shape;
    if (Array.isArray(inputShape)) {
      // If it's an array of shapes, use the first one
      // If it's just a simple shape array, use it directly
      if (Array.isArray(inputShape[0])) {
        shape = inputShape[0] as tf.Shape;
      } else {
        shape = inputShape as tf.Shape;
      }
    } else {
      shape = inputShape;
    }
    
    if (!shape || typeof shape === 'number' || shape.length < 2) {
      throw new Error('PositionalEncoding layer requires at least 2D input (sequence_length, embedding_dim)');
    }
    
    // Validate shape for transformer compatibility
    if (shape.length !== 2 && shape.length !== 3) {
      throw new Error(`PositionalEncoding layer expects 2D or 3D input ([sequence_length, embedding_dim] or [batch_size, sequence_length, embedding_dim]), but got shape with ${shape.length} dimensions`);
    }
    
    const embeddingDim = shape[shape.length - 1] as number;
    
    // Create positional encoding matrix
    // Create the encoding matrix
    const encodingData: number[][] = [];
    
    for (let pos = 0; pos < this.maxLength; pos++) {
      const row: number[] = [];
      for (let i = 0; i < embeddingDim; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / embeddingDim);
        if (i % 2 === 0) {
          row.push(Math.sin(angle));
        } else {
          row.push(Math.cos(angle));
        }
      }
      encodingData.push(row);
    }
    
    this.encodingMatrix = tf.tensor2d(encodingData, [this.maxLength, embeddingDim]);
    
    super.build(inputShape);
  }
  
  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const seqLength = input.shape[1]!;
      
      // Get the relevant part of the encoding matrix
      const encoding = tf.slice(this.encodingMatrix!, [0, 0], [seqLength, -1]);
      
      // Add positional encoding to input
      return tf.add(input, encoding);
    });
  }
  
  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }
  
  getConfig(): any {
    const config = super.getConfig();
    return {
      ...config,
      maxLength: this.maxLength
    };
  }
  
  static className = 'PositionalEncodingLayer';
}

/**
 * Transformer Encoder Block
 * Combines multi-head attention, normalization, and feed-forward network
 */
/**
 * Global Average Pooling 1D layer for sequence data.
 * Averages across the sequence dimension to produce fixed-size output.
 */
export class GlobalAveragePooling1DLayer extends LayerClass {
  constructor(config?: any) {
    super(config);
  }

  build(inputShape: tf.Shape | tf.Shape[]): void {
    // Handle both single shape and array of shapes
    let shape: tf.Shape;
    if (Array.isArray(inputShape)) {
      if (Array.isArray(inputShape[0])) {
        shape = inputShape[0] as tf.Shape;
      } else {
        shape = inputShape as tf.Shape;
      }
    } else {
      shape = inputShape;
    }
    
    // Validate that input shape represents sequence data
    // Accept both 2D ([sequence_length, features]) and 3D ([batch_size, sequence_length, features])
    if (!shape || shape.length < 2 || shape.length > 3) {
      throw new Error(`GlobalAveragePooling1D layer expects 2D or 3D input shape, but got shape with ${shape ? shape.length : 0} dimensions`);
    }
    
    super.build(inputShape);
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    const shape = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape as tf.Shape;
    // During build time: input shape is [sequence, features]
    // Output shape: [features] (batch dimension is implicit)
    if (shape.length === 2) {
      return [shape[1]] as tf.Shape;
    }
    // During runtime: input shape is [batch, sequence, features]
    // Output shape: [batch, features]
    return [shape[0], shape[2]] as tf.Shape;
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      
      // Validate tensor shape at runtime
      const inputRank = input.shape ? input.shape.length : 0;
      if (inputRank !== 3) {
        throw new Error(`GlobalAveragePooling1D expects 3D tensor input, but got tensor with rank ${inputRank}`);
      }
      
      // Average across the sequence dimension (axis=1)
      // For a 3D tensor, axis 1 is the sequence dimension
      return tf.mean(input, 1);
    });
  }

  getConfig(): any {
    const config = super.getConfig();
    return config;
  }

  static get className() {
    return 'GlobalAveragePooling1DLayer';
  }
}

export class TransformerEncoderBlock extends LayerClass {
  private numHeads: number;
  private keyDim: number;
  private ffDim: number;
  private dropout: number;
  
  private attention!: MultiHeadAttentionLayer;
  private ffn1!: tf.layers.Layer;
  private ffn2!: tf.layers.Layer;
  private layerNorm1!: tf.layers.Layer;
  private layerNorm2!: tf.layers.Layer;
  private dropout1!: tf.layers.Layer;
  private dropout2!: tf.layers.Layer;
  
  constructor(config: {
    numHeads: number;
    keyDim: number;
    ffDim: number;
    dropout?: number;
  }) {
    super({});
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.ffDim = config.ffDim;
    this.dropout = config.dropout || 0.1;
  }
  
  build(inputShape: tf.Shape | tf.Shape[]): void {
    // Handle both single shape and array of shapes
    let shape: tf.Shape;
    if (Array.isArray(inputShape)) {
      // If it's an array of shapes, use the first one
      // If it's just a simple shape array, use it directly
      if (Array.isArray(inputShape[0])) {
        shape = inputShape[0] as tf.Shape;
      } else {
        shape = inputShape as tf.Shape;
      }
    } else {
      shape = inputShape;
    }
    
    if (!shape || typeof shape === 'number' || shape.length < 2) {
      throw new Error('Invalid input shape');
    }
    const inputDim = shape[shape.length - 1] as number;
    
    // Create sub-layers
    this.attention = new MultiHeadAttentionLayer({
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      dropout: this.dropout
    });
    
    this.ffn1 = tf.layers.dense({
      units: this.ffDim,
      activation: 'relu'
    });
    
    this.ffn2 = tf.layers.dense({
      units: inputDim
    });
    
    this.layerNorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layerNorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    
    this.dropout1 = tf.layers.dropout({ rate: this.dropout });
    this.dropout2 = tf.layers.dropout({ rate: this.dropout });
    
    // Build sub-layers
    this.attention.build(shape);
    this.layerNorm1.build(shape);
    this.ffn1.build(shape);
    const ffnOutputShape = [...shape.slice(0, -1), this.ffDim] as tf.Shape;
    this.ffn2.build(ffnOutputShape);
    this.layerNorm2.build(shape);
    
    super.build(inputShape);
  }
  
  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const training = kwargs?.training || false;
      
      // Multi-head attention with residual connection
      let attentionOutput = this.attention.apply(input, { training }) as tf.Tensor;
      attentionOutput = this.dropout1.apply(attentionOutput, { training }) as tf.Tensor;
      const norm1 = this.layerNorm1.apply(tf.add(input, attentionOutput)) as tf.Tensor;
      
      // Feed-forward network with residual connection
      let ffnOutput = this.ffn1.apply(norm1) as tf.Tensor;
      ffnOutput = this.ffn2.apply(ffnOutput) as tf.Tensor;
      ffnOutput = this.dropout2.apply(ffnOutput, { training }) as tf.Tensor;
      const output = this.layerNorm2.apply(tf.add(norm1, ffnOutput)) as tf.Tensor;
      
      return output;
    });
  }
  
  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }
  
  getConfig(): any {
    const config = super.getConfig();
    return {
      ...config,
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      ffDim: this.ffDim,
      dropout: this.dropout
    };
  }
  
  static className = 'TransformerEncoderBlock';
}

// Register custom layers
tf.serialization.registerClass(MultiHeadAttentionLayer);
tf.serialization.registerClass(PositionalEncodingLayer);
tf.serialization.registerClass(TransformerEncoderBlock);
tf.serialization.registerClass(GlobalAveragePooling1DLayer);