<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';
	import type { ExperimentMetricColumn } from '$lib/research/metrics';

	let {
		experiments,
		onSelect,
		selected,
		sortMode = 'bpb',
		metricShortLabel = 'bpb',
		extraColumns = [],
		runNumberById = new Map<number, number>(),
		nextRunNumber = 1,
		selectionEnabled = false,
		selectedIds = [],
		onToggleBatchSelect
	}: {
		experiments: ExperimentRecord[];
		onSelect?: (exp: ExperimentRecord) => void;
		selected?: ExperimentRecord | null;
		sortMode?: 'bpb' | 'newest' | 'oldest' | 'steps' | 'name';
		metricShortLabel?: string;
		extraColumns?: ExperimentMetricColumn[];
		runNumberById?: Map<number, number>;
		nextRunNumber?: number;
		selectionEnabled?: boolean;
		selectedIds?: number[];
		onToggleBatchSelect?: (expId: number) => void;
	} = $props();

	let dataGridTemplate = $derived(
		['18px', '52px', 'minmax(12rem,1.4fr)', '18px', '56px', ...extraColumns.map((column) => column.width), '60px'].join(' ')
	);

	let sorted = $derived.by(() => {
		const items = [...experiments];
		switch (sortMode) {
			case 'name':
				return items.sort(
					(a, b) =>
						(a.name || '').localeCompare(b.name || '', undefined, { sensitivity: 'base' }) ||
						b.id - a.id
				);
			case 'newest':
				return items.sort((a, b) => {
					const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
					const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
					const byTime = (Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id);
					return byTime || b.id - a.id;
				});
			case 'oldest':
				return items.sort((a, b) => {
					const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
					const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
					const byTime = (Number.isFinite(aTime) ? aTime : a.id) - (Number.isFinite(bTime) ? bTime : b.id);
					return byTime || a.id - b.id;
				});
			case 'steps':
				return items.sort((a, b) => b.totalSteps - a.totalSteps || a.valBpb - b.valBpb);
			case 'bpb':
			default:
				return items.sort((a, b) => a.valBpb - b.valBpb || b.id - a.id);
		}
	});
	let bestId = $derived(
		[...experiments]
			.filter((exp) => exp.kept && !exp.error)
			.sort((a, b) => {
				const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
				const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
				return (Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id) || b.id - a.id;
			})[0]?.id
			?? (
				experiments.length > 0
					? [...experiments].sort((a, b) => a.valBpb - b.valBpb)[0].id
					: null
			)
	);
</script>

<div class="h-full overflow-auto">
	<div class="min-w-[28rem] space-y-0.5 pr-1">
		<div class="sticky top-0 z-10 flex items-center gap-1.5 border-b border-gray-800 bg-gray-950/95 px-1.5 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-gray-500 backdrop-blur">
			{#if selectionEnabled}
				<span class="ml-1 shrink-0 w-3" aria-hidden="true"></span>
			{/if}
			<div class="min-w-0 flex-1 grid items-center gap-1.5" style={`grid-template-columns: ${dataGridTemplate};`}>
				<span class="text-center" title="run source">src</span>
				<span class="tabular-nums text-right" title="experiment id">run</span>
				<span class="min-w-0" title="experiment name">experiment</span>
				<span class="text-center" title="rerun marker">r</span>
				<span class="tabular-nums text-right" title="training steps">steps</span>
				{#each extraColumns as column}
					<span class="tabular-nums text-right" title={column.title}>{column.label}</span>
				{/each}
				<span class="tabular-nums text-right" title="trainer validation score">{metricShortLabel}</span>
			</div>
		</div>
		{#each sorted as exp, i}
			<div
				class="w-full flex items-center gap-1.5 font-mono text-[11px] rounded transition-colors
					{exp.id === -1 ? 'bg-red-950/50 text-red-300' : selected?.id === exp.id ? 'bg-blue-950/50 text-blue-300' : exp.id === bestId ? 'bg-green-950/50 text-green-300' : 'text-gray-400 hover:bg-gray-800'}"
			>
				{#if selectionEnabled && exp.id !== -1}
					<button
						type="button"
						class="ml-1 shrink-0 w-3 h-3 rounded border border-gray-600 text-[9px] leading-none flex items-center justify-center {selectedIds.includes(exp.id) ? 'bg-blue-600 border-blue-500 text-white' : 'text-transparent'}"
						onclick={() => {
							onToggleBatchSelect?.(exp.id);
						}}
						title={selectedIds.includes(exp.id) ? 'remove from rerun selection' : 'add to rerun selection'}
					>
						✓
					</button>
				{/if}
				<button
					onclick={() => {
						if (exp.id < 1) return;
						onSelect?.(exp);
					}}
					class="min-w-0 flex-1 grid items-center gap-1.5 px-1.5 py-1 text-left"
					style={`grid-template-columns: ${dataGridTemplate};`}
				>
				{#if exp.id === -1}
					<span class="text-center text-red-400 animate-pulse" title="in progress">*</span>
				{:else}
					<span class="text-center {exp.source === 'auto' ? 'text-blue-400' : 'text-gray-500'}" title={exp.source === 'auto' ? 'automatic experiment' : 'manual experiment'}>
						{exp.source === 'auto' ? 'A' : 'M'}
					</span>
				{/if}
				{#if exp.id !== -1}
					<span class="tabular-nums text-[9px] text-right text-gray-500" title={`dataset run ${runNumberById.get(exp.id) ?? '?'} · db id ${exp.id}`}>
						#{runNumberById.get(exp.id) ?? '?'}
					</span>
				{:else}
					<span class="tabular-nums text-[9px] text-right text-gray-500" title={`next dataset run ${nextRunNumber}`}>
						#{nextRunNumber}
					</span>
				{/if}
				<span class="text-left flex-1 whitespace-nowrap pr-2" title={exp.reasoning}>
					{exp.name || `#${exp.id}`}
				</span>
				{#if exp.rerunOf}
					<span class="text-center text-[9px] text-amber-400" title={`rerun of #${exp.rerunOf}`}>R</span>
				{:else}
					<span class="text-center" aria-hidden="true"></span>
				{/if}
				<span class="tabular-nums text-[9px] text-right text-gray-500" title={`${exp.totalSteps} steps`}>
					{exp.id === -1 ? '...' : exp.totalSteps}
				</span>
				{#each extraColumns as column}
					<span class="tabular-nums text-right" title={column.title}>
						{column.render(column.value(exp))}
					</span>
				{/each}
				<span class="tabular-nums text-right">{exp.id === -1 && exp.valBpb === Infinity ? '...' : exp.valBpb.toFixed(3)}</span>
				</button>
			</div>
		{/each}

		{#if experiments.length === 0}
			<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
		{/if}
	</div>
</div>
