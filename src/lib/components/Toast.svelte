<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';

  export let type: 'success' | 'error' | 'info' = 'info';
  export let message: string;
  export let duration: number = 4000;
  export let autoClose: boolean = true;

  const dispatch = createEventDispatcher();
  let visible = false;

  onMount(() => {
    visible = true;
    
    if (autoClose && duration > 0) {
      setTimeout(() => {
        close();
      }, duration);
    }
  });

  function close() {
    visible = false;
    setTimeout(() => {
      dispatch('close');
    }, 300);
  }

  function getIcon() {
    switch (type) {
      case 'success': return '✓';
      case 'error': return '✕';
      case 'info': return 'ℹ';
      default: return 'ℹ';
    }
  }
</script>

<div 
  class="toast {type}" 
  class:visible 
  role="status"
  aria-live="polite"
>
  <div class="toast-content">
    <span class="toast-icon">{getIcon()}</span>
    <span class="toast-message">{message}</span>
    <button class="toast-close" on:click={close} aria-label="Close notification">✕</button>
  </div>
</div>

<style>
  .toast {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    min-width: 300px;
    max-width: 500px;
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    opacity: 0;
    transform: translateX(100%);
    transition: all 0.3s ease;
    cursor: pointer;
  }

  .toast.visible {
    opacity: 1;
    transform: translateX(0);
  }

  .toast.success {
    border-color: #16a34a;
    background: linear-gradient(to right, #052e16, #1a1a1a);
  }

  .toast.error {
    border-color: #dc2626;
    background: linear-gradient(to right, #450a0a, #1a1a1a);
  }

  .toast.info {
    border-color: #2563eb;
    background: linear-gradient(to right, #0c1e3e, #1a1a1a);
  }

  .toast-content {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
  }

  .toast-icon {
    font-size: 16px;
    font-weight: bold;
    color: inherit;
  }

  .toast.success .toast-icon {
    color: #22c55e;
  }

  .toast.error .toast-icon {
    color: #ef4444;
  }

  .toast.info .toast-icon {
    color: #3b82f6;
  }

  .toast-message {
    flex: 1;
    color: #ffffff;
    font-size: 14px;
    line-height: 1.4;
  }

  .toast-close {
    background: transparent;
    border: none;
    color: #737373;
    font-size: 14px;
    cursor: pointer;
    padding: 4px;
    line-height: 1;
    transition: color 0.2s;
  }

  .toast-close:hover {
    color: #ffffff;
  }
</style>