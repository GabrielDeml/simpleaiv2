import { describe, it, expect } from 'vitest';
import { ColabExporter } from './colabExporter';
import type { LayerConfig, TrainingConfig } from './types';

describe('ColabExporter', () => {
  const exporter = new ColabExporter();

  const basicTrainingConfig: TrainingConfig = {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    validationSplit: 0.2,
    loss: 'categoricalCrossentropy'
  };

  it('should generate a valid notebook structure', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', params: { shape: [28, 28] }, name: 'Input' },
      { id: '2', type: 'flatten', params: {}, name: 'Flatten' },
      { id: '3', type: 'dense', params: { units: 128, activation: 'relu', useBias: true }, name: 'Dense' },
      { id: '4', type: 'output', params: { units: 10, activation: 'softmax' }, name: 'Output' }
    ];

    const notebook = exporter.generateNotebook(layers, basicTrainingConfig, 'mnist');

    expect(notebook).toHaveProperty('cells');
    expect(notebook).toHaveProperty('metadata');
    expect(notebook).toHaveProperty('nbformat', 4);
    expect(notebook).toHaveProperty('nbformat_minor', 0);
    expect(notebook.cells).toHaveLength(5); // setup, dataset, model, training, evaluation
  });

  it('should handle transformer layers correctly', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', params: { shape: [100] }, name: 'Input' },
      { id: '2', type: 'embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 100, trainable: true }, name: 'Embedding' },
      { id: '3', type: 'positionalEncoding', params: { maxLength: 100 }, name: 'Positional Encoding' },
      { id: '4', type: 'multiHeadAttention', params: { numHeads: 8, keyDim: 64, valueDim: 64, dropout: 0.1, useBias: true }, name: 'Multi-Head Attention' },
      { id: '5', type: 'layerNormalization', params: { epsilon: 1e-6, center: true, scale: true }, name: 'Layer Norm' },
      { id: '6', type: 'transformerBlock', params: { numHeads: 8, keyDim: 64, ffDim: 256, dropout: 0.1 }, name: 'Transformer Block' },
      { id: '7', type: 'flatten', params: {}, name: 'Flatten' },
      { id: '8', type: 'output', params: { units: 4, activation: 'softmax' }, name: 'Output' }
    ];

    const notebook = exporter.generateNotebook(layers, basicTrainingConfig, 'ag-news');
    
    // Check that the notebook was generated
    expect(notebook.cells).toHaveLength(5);
    
    // Check setup cell contains custom transformer layer implementations
    const setupCell = notebook.cells[0];
    const setupCode = setupCell.source.join('');
    expect(setupCode).toContain('class PositionalEncoding(layers.Layer)');
    expect(setupCode).toContain('class TransformerBlock(layers.Layer)');
    
    // Check model cell contains transformer layers
    const modelCell = notebook.cells[2];
    const modelCode = modelCell.source.join('');
    
    expect(modelCode).toContain('layers.Embedding(input_dim=10000, output_dim=128, input_length=100, trainable=True)');
    expect(modelCode).toContain('PositionalEncoding(max_length=100)');
    expect(modelCode).toContain('layers.MultiHeadAttention(num_heads=8, key_dim=64, value_dim=64, dropout=0.1, use_bias=True)');
    expect(modelCode).toContain('layers.LayerNormalization(epsilon=0.000001, center=True, scale=True)');
    expect(modelCode).toContain('TransformerBlock(num_heads=8, key_dim=64, ff_dim=256, dropout=0.1)');
  });

  it('should handle different datasets correctly', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', params: { shape: [28, 28] }, name: 'Input' },
      { id: '2', type: 'output', params: { units: 10, activation: 'softmax' }, name: 'Output' }
    ];

    // Test MNIST
    const mnistNotebook = exporter.generateNotebook(layers, basicTrainingConfig, 'mnist');
    const mnistDataCell = mnistNotebook.cells[1];
    const mnistCode = mnistDataCell.source.join('');
    expect(mnistCode).toContain('keras.datasets.mnist');
    expect(mnistCode).toContain('28x28 grayscale');

    // Test IMDB
    const imdbNotebook = exporter.generateNotebook(layers, basicTrainingConfig, 'imdb');
    const imdbDataCell = imdbNotebook.cells[1];
    const imdbCode = imdbDataCell.source.join('');
    expect(imdbCode).toContain('imdb.load_data');
    expect(imdbCode).toContain('pad_sequences');

    // Test AG News
    const agNewsNotebook = exporter.generateNotebook(layers, basicTrainingConfig, 'ag-news');
    const agNewsDataCell = agNewsNotebook.cells[1];
    const agNewsCode = agNewsDataCell.source.join('');
    expect(agNewsCode).toContain('ag_news_subset');
    expect(agNewsCode).toContain('Tokenizer');
  });

  it('should generate correct filenames', () => {
    const mnistFilename = exporter.generateFilename('mnist');
    expect(mnistFilename).toMatch(/^neural-network-mnist-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.ipynb$/);

    const imdbFilename = exporter.generateFilename('imdb');
    expect(imdbFilename).toMatch(/^neural-network-imdb-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.ipynb$/);
  });

  it.skip('should generate download links (requires browser environment)', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', params: { shape: [28, 28] }, name: 'Input' },
      { id: '2', type: 'output', params: { units: 10, activation: 'softmax' }, name: 'Output' }
    ];

    const downloadUrl = exporter.generateDownloadLink(layers, basicTrainingConfig, 'mnist');
    expect(downloadUrl).toMatch(/^blob:/);
  });

  it('should convert layer parameters to Python format correctly', () => {
    const layers: LayerConfig[] = [
      { id: '1', type: 'input', params: { shape: [28, 28] }, name: 'Input' },
      { id: '2', type: 'dense', params: { units: 128, activation: 'relu', useBias: false }, name: 'Dense' },
      { id: '3', type: 'conv2d', params: { filters: 32, kernelSize: [3, 3], strides: [1, 1], padding: 'same', activation: 'relu', useBias: true }, name: 'Conv2D' }
    ];

    const notebook = exporter.generateNotebook(layers, basicTrainingConfig, 'mnist');
    const modelCell = notebook.cells[2];
    const modelCode = modelCell.source.join('');
    
    // Check boolean conversion
    expect(modelCode).toContain('use_bias=False');
    expect(modelCode).toContain('use_bias=True');
    
    // Check array conversion
    expect(modelCode).toContain('(3, 3)');
    expect(modelCode).toContain('(1, 1)');
  });
});