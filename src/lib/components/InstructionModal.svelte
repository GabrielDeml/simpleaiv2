<script lang="ts">
  export let title: string;
  export let instructions: string[];
  export let showCloseButton: boolean = true;

  // Event callback (Svelte 5 style)
  export let onclose: (() => void) | undefined = undefined;

  function handleClose() {
    onclose?.();
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      handleClose();
    }
  }
</script>

<div 
  class="modal-backdrop" 
  on:click={handleBackdropClick}
  on:keydown={(e) => e.key === 'Escape' && handleClose()}
  role="dialog"
  aria-modal="true"
  aria-labelledby="instruction-modal-title"
  tabindex="-1"
>
  <div class="instruction-modal">
    <div class="modal-header">
      <h2 id="instruction-modal-title" class="modal-title">{title}</h2>
      {#if showCloseButton}
        <button class="close-button" on:click={handleClose}>✕</button>
      {/if}
    </div>
    
    <div class="modal-body">
      <div class="instructions">
        {#each instructions as instruction, index}
          <div class="instruction-item">
            <span class="step-number">{index + 1}</span>
            <span class="instruction-text">{instruction}</span>
          </div>
        {/each}
      </div>
    </div>
    
    <div class="modal-footer">
      <button class="btn btn-primary" on:click={handleClose}>
        Got it
      </button>
    </div>
  </div>
</div>

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .instruction-modal {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    overflow: hidden;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-20px) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px 24px 0 24px;
    border-bottom: 1px solid #333333;
    padding-bottom: 16px;
    margin-bottom: 24px;
  }

  .modal-title {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
  }

  .close-button {
    background: transparent;
    border: none;
    color: #737373;
    font-size: 18px;
    cursor: pointer;
    padding: 4px;
    line-height: 1;
    transition: color 0.2s;
  }

  .close-button:hover {
    color: #ffffff;
  }

  .modal-body {
    padding: 0 24px;
    max-height: 400px;
    overflow-y: auto;
  }

  .instructions {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .instruction-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .step-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: #2563eb;
    color: white;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .instruction-text {
    font-size: 14px;
    line-height: 1.5;
    color: #d4d4d4;
    flex: 1;
  }

  .modal-footer {
    padding: 24px;
    display: flex;
    justify-content: flex-end;
    border-top: 1px solid #333333;
    margin-top: 24px;
  }

  .btn {
    padding: 10px 24px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-primary {
    background: #2563eb;
    color: white;
  }

  .btn-primary:hover {
    background: #1d4ed8;
  }
</style>