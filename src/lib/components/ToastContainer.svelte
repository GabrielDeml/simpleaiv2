<script lang="ts">
  import { toasts, removeToast } from '$lib/stores/toastStore';
  import Toast from './Toast.svelte';
</script>

<div class="toast-container">
  {#each $toasts as toast (toast.id)}
    <Toast 
      type={toast.type}
      message={toast.message}
      duration={toast.duration}
      autoClose={toast.autoClose}
      onclose={() => removeToast(toast.id)}
    />
  {/each}
</div>

<style>
  .toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
  }

  .toast-container :global(.toast) {
    pointer-events: auto;
    position: relative;
    top: auto;
    right: auto;
  }
</style>