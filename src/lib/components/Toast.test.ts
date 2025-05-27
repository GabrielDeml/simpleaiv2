import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/svelte';
import Toast from './Toast.svelte';

describe('Toast', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it('renders with message', () => {
    render(Toast, { props: { message: 'Test message' } });
    expect(screen.getByText('Test message')).toBeInTheDocument();
  });

  it('renders with correct type styling', () => {
    const { container: successContainer } = render(Toast, { 
      props: { message: 'Success', type: 'success' } 
    });
    expect(successContainer.querySelector('.toast.success')).toBeInTheDocument();

    const { container: errorContainer } = render(Toast, { 
      props: { message: 'Error', type: 'error' } 
    });
    expect(errorContainer.querySelector('.toast.error')).toBeInTheDocument();

    const { container: infoContainer } = render(Toast, { 
      props: { message: 'Info', type: 'info' } 
    });
    expect(infoContainer.querySelector('.toast.info')).toBeInTheDocument();
  });

  it('displays correct icon based on type', () => {
    const { container } = render(Toast, { 
      props: { message: 'Test', type: 'success' } 
    });
    expect(container.querySelector('.toast-icon')?.textContent).toBe('✓');

    cleanup();
    const { container: errorContainer } = render(Toast, { 
      props: { message: 'Test', type: 'error' } 
    });
    expect(errorContainer.querySelector('.toast-icon')?.textContent).toBe('✕');

    cleanup();
    const { container: infoContainer } = render(Toast, { 
      props: { message: 'Test', type: 'info' } 
    });
    expect(infoContainer.querySelector('.toast-icon')?.textContent).toBe('ℹ');
  });

  it('becomes visible after mount', async () => {
    const { container } = render(Toast, { props: { message: 'Test' } });
    const toast = container.querySelector('.toast');
    
    await waitFor(() => {
      expect(toast).toHaveClass('visible');
    });
  });

  it('auto-closes after duration', async () => {
    const { container } = render(Toast, { 
      props: { 
        message: 'Test', 
        duration: 1000, 
        autoClose: true
      } 
    });
    
    const toast = container.querySelector('.toast');
    
    // Wait for mount
    await vi.waitFor(() => {
      expect(toast).toHaveClass('visible');
    });
    
    // Fast forward time for duration + animation
    await vi.advanceTimersToNextTimerAsync();
    await vi.advanceTimersToNextTimerAsync();
    
    // Check that toast is no longer visible
    expect(toast).not.toHaveClass('visible');
  });

  it('does not auto-close when autoClose is false', async () => {
    const { container } = render(Toast, { 
      props: { 
        message: 'Test', 
        duration: 1000, 
        autoClose: false
      } 
    });
    
    const toast = container.querySelector('.toast');
    
    // Wait for mount
    await waitFor(() => {
      expect(toast).toHaveClass('visible');
    });
    
    vi.advanceTimersByTime(2000);
    
    // Should still be visible
    expect(toast).toHaveClass('visible');
  });

  it('closes on click', async () => {
    const { container } = render(Toast, { 
      props: { 
        message: 'Test', 
        autoClose: false
      } 
    });
    
    const toast = container.querySelector('.toast');
    
    // Wait for mount
    await vi.waitFor(() => {
      expect(toast).toHaveClass('visible');
    });
    
    await fireEvent.click(toast!);
    
    // Wait for close animation
    await vi.advanceTimersToNextTimerAsync();
    
    // Check that toast is no longer visible
    expect(toast).not.toHaveClass('visible');
  });

  it('closes on close button click', async () => {
    const { container } = render(Toast, { 
      props: { 
        message: 'Test', 
        autoClose: false
      } 
    });
    
    const toast = container.querySelector('.toast');
    
    // Wait for mount
    await vi.waitFor(() => {
      expect(toast).toHaveClass('visible');
    });
    
    const closeButton = screen.getByRole('button');
    await fireEvent.click(closeButton);
    
    // Wait for close animation
    await vi.advanceTimersToNextTimerAsync();
    
    // Check that toast is no longer visible
    expect(toast).not.toHaveClass('visible');
  });
});