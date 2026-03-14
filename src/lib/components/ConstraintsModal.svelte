<script lang="ts">
	import type { ParamConstraints } from '$lib/model/config';

	let {
		constraints = $bindable(),
		onClose
	}: {
		constraints: ParamConstraints;
		onClose: () => void;
	} = $props();

	const fields: { key: keyof ParamConstraints; label: string; step: number }[] = [
		{ key: 'nLayer', label: 'layers', step: 1 },
		{ key: 'nEmbd', label: 'd_model', step: 32 },
		{ key: 'nHead', label: 'heads', step: 1 },
		{ key: 'mlpRatio', label: 'mlp_ratio', step: 1 },
		{ key: 'lr', label: 'lr', step: 0.0001 },
		{ key: 'weightDecay', label: 'weight_decay', step: 0.01 },
		{ key: 'warmupRatio', label: 'warmup_ratio', step: 0.05 },
		{ key: 'cooldownRatio', label: 'cooldown_ratio', step: 0.05 },
		{ key: 'batchSize', label: 'batch_size', step: 1 },
		{ key: 'seqLen', label: 'seq_len', step: 32 },
		{ key: 'trainSeconds', label: 'train_sec', step: 10 },
		{ key: 'maxParams', label: 'max params', step: 100000 },
	];
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onclick={onClose}>
	<div class="bg-gray-900 border border-gray-700 rounded-lg p-5 w-[540px] space-y-4 font-mono" onclick={(e: MouseEvent) => e.stopPropagation()}>
		<h3 class="text-sm text-gray-200">constraints</h3>
		<p class="text-xs text-gray-500">Set min/max bounds for auto research. Leave blank for no constraint.</p>

		<div class="space-y-1.5 max-h-80 overflow-y-auto">
			<div class="grid grid-cols-[1fr_120px_120px] gap-2 text-xs text-gray-500 mb-1">
				<span></span>
				<span class="text-center">min</span>
				<span class="text-center">max</span>
			</div>
			{#each fields as f}
				<div class="grid grid-cols-[1fr_120px_120px] gap-2 items-center">
					<span class="text-xs text-gray-400">{f.label}</span>
					<input
						type="number"
						value={constraints[f.key]?.min ?? ''}
						oninput={(e: Event) => {
							const v = (e.target as HTMLInputElement).value;
							if (!constraints[f.key]) constraints[f.key] = {};
							constraints[f.key]!.min = v === '' ? undefined : Number(v);
						}}
						step={f.step}
						placeholder="—"
						class="bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-gray-200 text-right tabular-nums placeholder-gray-600"
					/>
					<input
						type="number"
						value={constraints[f.key]?.max ?? ''}
						oninput={(e: Event) => {
							const v = (e.target as HTMLInputElement).value;
							if (!constraints[f.key]) constraints[f.key] = {};
							constraints[f.key]!.max = v === '' ? undefined : Number(v);
						}}
						step={f.step}
						placeholder="—"
						class="bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-gray-200 text-right tabular-nums placeholder-gray-600"
					/>
				</div>
			{/each}
		</div>

		<div class="flex justify-end">
			<button
				onclick={onClose}
				class="px-3 py-1.5 rounded bg-gray-700 text-gray-200 hover:bg-gray-600 text-sm"
			>done</button>
		</div>
	</div>
</div>
