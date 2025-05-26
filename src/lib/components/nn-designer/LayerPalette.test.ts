import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import LayerPalette from './LayerPalette.svelte';
import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
import { addLayer } from '$lib/nn-designer/stores';

// Mock the stores module
vi.mock('$lib/nn-designer/stores', () => ({
  addLayer: vi.fn()
}));

describe('LayerPalette', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all layer types from layerDefinitions', () => {
    render(LayerPalette);
    
    Object.values(layerDefinitions).forEach((definition) => {
      expect(screen.getByText(definition.displayName)).toBeInTheDocument();
    });
  });

  it('renders layer icons', () => {
    render(LayerPalette);
    
    Object.values(layerDefinitions).forEach((definition) => {
      expect(screen.getByText(definition.icon)).toBeInTheDocument();
    });
  });

  it('collapses and expands when header is clicked', async () => {
    const { container } = render(LayerPalette);
    
    // Initially expanded
    expect(container.querySelector('.layer-cards')).toBeInTheDocument();
    
    // Click to collapse
    const header = screen.getByText('LAYERS').parentElement!;
    await fireEvent.click(header);
    
    // Should be collapsed
    expect(container.querySelector('.layer-cards')).not.toBeInTheDocument();
    
    // Click to expand
    await fireEvent.click(header);
    
    // Should be expanded again
    expect(container.querySelector('.layer-cards')).toBeInTheDocument();
  });

  it('adds layer when card is clicked', async () => {
    render(LayerPalette);
    
    const denseCard = screen.getByText('Dense').parentElement!;
    await fireEvent.click(denseCard);
    
    expect(addLayer).toHaveBeenCalledWith({
      id: expect.stringMatching(/^dense-\d+$/),
      type: 'dense',
      name: 'Dense',
      params: expect.objectContaining(layerDefinitions.dense.defaultParams)
    });
  });

  it('adds layer with keyboard interaction', async () => {
    render(LayerPalette);
    
    const denseCard = screen.getByText('Dense').parentElement!;
    
    // Test Enter key
    await fireEvent.keyDown(denseCard, { key: 'Enter' });
    expect(addLayer).toHaveBeenCalledTimes(1);
    
    // Test Space key
    await fireEvent.keyDown(denseCard, { key: ' ' });
    expect(addLayer).toHaveBeenCalledTimes(2);
  });

  it('sets drag data on drag start', async () => {
    render(LayerPalette);
    
    const denseCard = screen.getByText('Dense').parentElement!;
    const mockDataTransfer = {
      effectAllowed: '',
      setData: vi.fn()
    };
    
    await fireEvent.dragStart(denseCard, {
      dataTransfer: mockDataTransfer as any
    });
    
    expect(mockDataTransfer.effectAllowed).toBe('copy');
    expect(mockDataTransfer.setData).toHaveBeenCalledWith('layerType', 'dense');
  });

  it('has draggable attribute on layer cards', () => {
    const { container } = render(LayerPalette);
    
    const layerCards = container.querySelectorAll('.layer-card');
    layerCards.forEach(card => {
      expect(card).toHaveAttribute('draggable', 'true');
    });
  });

  it('has proper ARIA attributes for accessibility', () => {
    const { container } = render(LayerPalette);
    
    const layerCards = container.querySelectorAll('.layer-card');
    layerCards.forEach(card => {
      expect(card).toHaveAttribute('role', 'button');
      expect(card).toHaveAttribute('tabindex', '0');
    });
  });

  it('generates unique layer IDs', async () => {
    render(LayerPalette);
    
    const denseCard = screen.getByText('Dense').parentElement!;
    
    // Add small delay between clicks to ensure different timestamps
    await fireEvent.click(denseCard);
    await new Promise(resolve => setTimeout(resolve, 10));
    await fireEvent.click(denseCard);
    
    expect(addLayer).toHaveBeenCalledTimes(2);
    
    const firstCall = (addLayer as any).mock.calls[0][0];
    const secondCall = (addLayer as any).mock.calls[1][0];
    
    expect(firstCall.id).not.toBe(secondCall.id);
    expect(firstCall.id).toMatch(/^dense-\d+$/);
    expect(secondCall.id).toMatch(/^dense-\d+$/);
  });

  it('creates independent parameter objects for each layer', async () => {
    render(LayerPalette);
    
    const denseCard = screen.getByText('Dense').parentElement!;
    
    await fireEvent.click(denseCard);
    await fireEvent.click(denseCard);
    
    const firstParams = (addLayer as any).mock.calls[0][0].params;
    const secondParams = (addLayer as any).mock.calls[1][0].params;
    
    // Should be different objects
    expect(firstParams).not.toBe(secondParams);
    // But with same values
    expect(firstParams).toEqual(secondParams);
  });

  it('info buttons have proper aria labels', () => {
    render(LayerPalette);
    
    Object.values(layerDefinitions).forEach((definition) => {
      const infoButton = screen.getByLabelText(`Show ${definition.displayName} details`);
      expect(infoButton).toBeInTheDocument();
    });
  });

  it('stops propagation when info button is clicked', async () => {
    render(LayerPalette);
    
    const infoButton = screen.getByLabelText('Show Dense details');
    await fireEvent.click(infoButton);
    
    // addLayer should not have been called
    expect(addLayer).not.toHaveBeenCalled();
  });
});