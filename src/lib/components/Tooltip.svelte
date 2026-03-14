<script lang="ts">
	import type { Snippet } from 'svelte';

	let { text, children }: { text: string; children: Snippet } = $props();

	let show = $state(false);
	let x = $state(0);
	let y = $state(0);
	let el: HTMLSpanElement;

	function onEnter(e: MouseEvent) {
		show = true;
		updatePos(e);
	}

	function onMove(e: MouseEvent) {
		updatePos(e);
	}

	function onLeave() {
		show = false;
	}

	function updatePos(e: MouseEvent) {
		x = e.clientX;
		y = e.clientY;
	}
</script>

<span
	bind:this={el}
	onmouseenter={onEnter}
	onmousemove={onMove}
	onmouseleave={onLeave}
	class="inline"
>
	{@render children()}
</span>

{#if show}
	<div
		class="fixed z-50 max-w-56 px-2.5 py-1.5 rounded bg-gray-800 border border-gray-700 text-[11px] font-mono text-gray-300 leading-relaxed shadow-lg pointer-events-none"
		style="left: {x + 12}px; top: {y - 8}px; transform: translateY(-100%);"
	>
		{text}
	</div>
{/if}
