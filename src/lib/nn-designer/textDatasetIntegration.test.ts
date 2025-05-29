import { describe, it, expect } from 'vitest';
import { ModelBuilder } from './modelBuilder';
import { colabExporter } from './colabExporter';
import type { LayerConfig, TrainingConfig } from './types';

describe('Text Dataset and Transformer Integration', () => {
  const trainingConfig: TrainingConfig = {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    validationSplit: 0.2,
    loss: 'categoricalCrossentropy'
  };

  it('should build a complete transformer model for text classification', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', name: 'Input', params: { shape: [200] } },
      { id: '2', type: 'embedding', name: 'Embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 200, trainable: true } },
      { id: '3', type: 'positionalEncoding', name: 'Positional Encoding', params: { maxLength: 200 } },
      { id: '4', type: 'transformerBlock', name: 'Transformer Block 1', params: { numHeads: 8, keyDim: 16, ffDim: 128, dropout: 0.1 } },
      { id: '5', type: 'transformerBlock', name: 'Transformer Block 2', params: { numHeads: 8, keyDim: 16, ffDim: 128, dropout: 0.1 } },
      { id: '6', type: 'globalAveragePooling1D', name: 'Global Average Pooling', params: {} },
      { id: '7', type: 'dense', name: 'Dense', params: { units: 64, activation: 'relu', useBias: true } },
      { id: '8', type: 'dropout', name: 'Dropout', params: { rate: 0.3 } },
      { id: '9', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
    ];

    const builder = new ModelBuilder();
    const model = builder.buildModel(layers);

    // Verify model was built successfully
    expect(model).toBeDefined();
    expect(model.layers).toHaveLength(8); // All layers except input

    // Check that the model compiles successfully
    expect(() => builder.compileModel(trainingConfig)).not.toThrow();
  });

  it('should export transformer model to Colab with proper custom layers', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', name: 'Input', params: { shape: [256] } },
      { id: '2', type: 'embedding', name: 'Embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 256, trainable: true } },
      { id: '3', type: 'positionalEncoding', name: 'Positional Encoding', params: { maxLength: 256 } },
      { id: '4', type: 'transformerBlock', name: 'Transformer Block', params: { numHeads: 8, keyDim: 32, ffDim: 128, dropout: 0.2 } },
      { id: '5', type: 'globalAveragePooling1D', name: 'Global Average Pooling', params: {} },
      { id: '6', type: 'dense', name: 'Dense', params: { units: 64, activation: 'relu', useBias: true } },
      { id: '7', type: 'dropout', name: 'Dropout', params: { rate: 0.5 } },
      { id: '8', type: 'output', name: 'Output', params: { units: 4, activation: 'softmax' } }
    ];

    const notebook = colabExporter.generateNotebook(layers, trainingConfig, 'ag-news');

    // Check setup cell includes custom layer definitions
    const setupCode = notebook.cells[0].source.join('');
    expect(setupCode).toContain('class PositionalEncoding(layers.Layer)');
    expect(setupCode).toContain('class TransformerBlock(layers.Layer)');

    // Check model cell uses custom layers correctly
    const modelCode = notebook.cells[2].source.join('');
    expect(modelCode).toContain('keras.Input(shape=[256])');
    expect(modelCode).toContain('layers.Embedding(input_dim=10000, output_dim=128, input_length=256, trainable=True)');
    expect(modelCode).toContain('PositionalEncoding(max_length=256)');
    expect(modelCode).toContain('TransformerBlock(num_heads=8, key_dim=32, ff_dim=128, dropout=0.2)');
    expect(modelCode).toContain('layers.GlobalAveragePooling1D()');
    expect(modelCode).toContain('layers.Dense(64, activation=\'relu\', use_bias=True)');
    expect(modelCode).toContain('layers.Dropout(0.5)');
    expect(modelCode).toContain('layers.Dense(4, activation=\'softmax\')');

    // Check dataset handling for AG News
    const datasetCode = notebook.cells[1].source.join('');
    expect(datasetCode).toContain('ag_news_subset');
    expect(datasetCode).toContain('vocab_size = 10000');
    expect(datasetCode).toContain('max_length = 120');
    expect(datasetCode).toContain('Tokenizer(num_words=vocab_size');
    expect(datasetCode).toContain('pad_sequences');
  });

  it('should support different text model architectures', () => {
    // Simple embedding + dense model
    const simpleLayers: LayerConfig[] = [
      { id: '1', type: 'input', name: 'Input', params: { shape: [100] } },
      { id: '2', type: 'embedding', name: 'Embedding', params: { vocabSize: 5000, embeddingDim: 50, maxLength: 100 } },
      { id: '3', type: 'globalAveragePooling1D', name: 'Global Average Pooling', params: {} },
      { id: '4', type: 'output', name: 'Output', params: { units: 2, activation: 'sigmoid' } }
    ];

    const simpleBuilder = new ModelBuilder();
    const simpleModel = simpleBuilder.buildModel(simpleLayers);
    expect(simpleModel).toBeDefined();
    expect(simpleModel.layers).toHaveLength(3);
    expect(() => simpleBuilder.compileModel(trainingConfig)).not.toThrow();

    // Complex multi-head attention model
    const complexLayers: LayerConfig[] = [
      { id: '1', type: 'input', name: 'Input', params: { shape: [512] } },
      { id: '2', type: 'embedding', name: 'Embedding', params: { vocabSize: 20000, embeddingDim: 256, maxLength: 512 } },
      { id: '3', type: 'multiHeadAttention', name: 'Multi-Head Attention', params: { numHeads: 16, keyDim: 64, dropout: 0.1 } },
      { id: '4', type: 'layerNormalization', name: 'Layer Norm', params: {} },
      { id: '5', type: 'globalAveragePooling1D', name: 'Global Average Pooling', params: {} },
      { id: '6', type: 'output', name: 'Output', params: { units: 10, activation: 'softmax' } }
    ];

    const complexBuilder = new ModelBuilder();
    const complexModel = complexBuilder.buildModel(complexLayers);
    expect(complexModel).toBeDefined();
    expect(complexModel.layers).toHaveLength(5);
    expect(() => complexBuilder.compileModel(trainingConfig)).not.toThrow();
  });

  it('should handle IMDB dataset configuration in Colab export', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', name: 'Input', params: { shape: [256] } },
      { id: '2', type: 'embedding', name: 'Embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 256 } },
      { id: '3', type: 'transformerBlock', name: 'Transformer', params: { numHeads: 8, keyDim: 16, ffDim: 128, dropout: 0.1 } },
      { id: '4', type: 'globalAveragePooling1D', name: 'Global Average Pooling', params: {} },
      { id: '5', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
    ];

    const notebook = colabExporter.generateNotebook(layers, trainingConfig, 'imdb');
    const datasetCode = notebook.cells[1].source.join('');

    expect(datasetCode).toContain('imdb.load_data(num_words=vocab_size)');
    expect(datasetCode).toContain('vocab_size = 10000');
    expect(datasetCode).toContain('max_length = 256');
    expect(datasetCode).toContain('sequence.pad_sequences');
    expect(datasetCode).toContain("class_names = ['Negative', 'Positive']");
  });
});