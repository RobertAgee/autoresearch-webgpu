<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';
	import { getExperimentPrimaryMetric } from '$lib/research/metrics';

	type ScopeMode = 'all' | 'adHoc' | 'reruns' | 'benchmark' | 'family';

	type SummaryRow = {
		key: string;
		label: string;
		runCount: number;
		bestValBpb: number | null;
		avgValBpb: number | null;
		errorCount: number;
		groupCount: number;
		latestId: number;
	};

	let {
		experiments,
		scopeMode,
		metricShortLabel = 'score',
		recipeKey = null,
		selectedBenchmarkGroup = null,
		selectedModelFamily = null,
		onSelectAll,
		onSelectAdHoc,
		onSelectReruns,
		onSelectBenchmarkGroup,
		onSelectModelFamily
	}: {
		experiments: ExperimentRecord[];
		scopeMode: ScopeMode;
		metricShortLabel?: string;
		recipeKey?: string | null;
		selectedBenchmarkGroup?: string | null;
		selectedModelFamily?: string | null;
		onSelectAll?: () => void;
		onSelectAdHoc?: () => void;
		onSelectReruns?: () => void;
		onSelectBenchmarkGroup?: (group: string) => void;
		onSelectModelFamily?: (family: string) => void;
	} = $props();

	function experimentTimeValue(exp: ExperimentRecord): number {
		if (!exp.createdAt) return exp.id;
		const parsed = Date.parse(exp.createdAt);
		return Number.isFinite(parsed) ? parsed : exp.id;
	}

	function formatMetric(value: number | null): string {
		if (value == null) return '--';
		return recipeKey?.includes('abc') ? value.toFixed(1) : value.toFixed(3);
	}

	function summarizeRows(entries: Iterable<[string, ExperimentRecord[]]>, groupCountFor: (runs: ExperimentRecord[]) => number): SummaryRow[] {
		return [...entries]
			.map(([key, runs]) => {
				const valid = runs
					.filter((exp) => !exp.error)
					.map((exp) => getExperimentPrimaryMetric(exp).value)
					.filter((value): value is number => value != null && Number.isFinite(value));
				const bestValBpb = valid.length > 0
					? (recipeKey?.includes('abc') ? Math.max(...valid) : Math.min(...valid))
					: null;
				const avgValBpb = valid.length > 0
					? valid.reduce((sum, value) => sum + value, 0) / valid.length
					: null;
				const latest = [...runs].sort((a, b) => experimentTimeValue(b) - experimentTimeValue(a) || b.id - a.id)[0];
				return {
					key,
					label: key,
					runCount: runs.length,
					bestValBpb,
					avgValBpb,
					errorCount: runs.filter((exp) => Boolean(exp.error)).length,
					groupCount: groupCountFor(runs),
					latestId: latest?.id ?? 0
				};
			})
			.sort((a, b) => {
				const bestDelta = recipeKey?.includes('abc')
					? (b.bestValBpb ?? Number.NEGATIVE_INFINITY) - (a.bestValBpb ?? Number.NEGATIVE_INFINITY)
					: (a.bestValBpb ?? Number.POSITIVE_INFINITY) - (b.bestValBpb ?? Number.POSITIVE_INFINITY);
				if (bestDelta !== 0) return bestDelta;
				return b.latestId - a.latestId || b.runCount - a.runCount || a.label.localeCompare(b.label);
			});
	}

	let adHocRunCount = $derived(
		experiments.filter((exp) => !exp.benchmarkGroup).length
	);
	let rerunCount = $derived(
		experiments.filter((exp) => Boolean(exp.rerunOf)).length
	);
	let benchmarkRows = $derived.by(() => {
		const groups = new Map<string, ExperimentRecord[]>();
		for (const exp of experiments) {
			if (!exp.benchmarkGroup) continue;
			const bucket = groups.get(exp.benchmarkGroup) ?? [];
			bucket.push(exp);
			groups.set(exp.benchmarkGroup, bucket);
		}
		return summarizeRows(groups.entries(), (runs) => new Set(runs.map((exp) => exp.rerunOf).filter((value): value is number => value != null)).size);
	});
	let familyRows = $derived.by(() => {
		const groups = new Map<string, ExperimentRecord[]>();
		for (const exp of experiments) {
			const family = exp.modelFamily?.trim() || 'byte-gpt';
			const bucket = groups.get(family) ?? [];
			bucket.push(exp);
			groups.set(family, bucket);
		}
		return summarizeRows(groups.entries(), (runs) => new Set(runs.map((exp) => exp.benchmarkGroup).filter(Boolean)).size);
	});
</script>

<div class="space-y-3 rounded border border-gray-800 bg-black/20 p-2">
	<div class="space-y-1">
		<p class="text-[10px] font-mono uppercase tracking-[0.14em] text-gray-500">comparison scope</p>
		<div class="flex flex-wrap gap-1.5 text-[10px] font-mono">
			<button
				type="button"
				onclick={() => onSelectAll?.()}
				class="rounded border px-2 py-1 transition-colors {scopeMode === 'all' ? 'border-blue-500 bg-blue-950/50 text-blue-200' : 'border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'}"
			>
				all runs ({experiments.length})
			</button>
			<button
				type="button"
				onclick={() => onSelectAdHoc?.()}
				class="rounded border px-2 py-1 transition-colors {scopeMode === 'adHoc' ? 'border-emerald-500 bg-emerald-950/40 text-emerald-200' : 'border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'}"
			>
				ad hoc ({adHocRunCount})
			</button>
			<button
				type="button"
				onclick={() => onSelectReruns?.()}
				class="rounded border px-2 py-1 transition-colors {scopeMode === 'reruns' ? 'border-amber-500 bg-amber-950/40 text-amber-200' : 'border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'}"
			>
				reruns ({rerunCount})
			</button>
		</div>
		<p class="text-[10px] font-mono text-gray-600">
			Benchmark gates decide which runs are kept. Validation score is still shown, but it is no longer the only truth.
		</p>
	</div>

	<div class="space-y-1">
		<div class="flex items-center justify-between">
			<p class="text-[10px] font-mono uppercase tracking-[0.14em] text-gray-500">benchmark cohorts</p>
			<span class="text-[10px] font-mono text-gray-600">{benchmarkRows.length} groups</span>
		</div>
		{#if benchmarkRows.length > 0}
			<div class="rounded border border-gray-800 overflow-hidden">
				<div class="grid grid-cols-[minmax(0,1fr)_44px_44px_44px] gap-2 border-b border-gray-800 bg-gray-950/80 px-2 py-1 text-[9px] font-mono uppercase tracking-[0.14em] text-gray-500">
					<span>group</span>
					<span class="text-right">runs</span>
					<span class="text-right">best</span>
					<span class="text-right">avg</span>
				</div>
				<div class="max-h-36 overflow-y-auto">
					{#each benchmarkRows as row}
						<button
							type="button"
							onclick={() => onSelectBenchmarkGroup?.(row.key)}
							class="grid w-full grid-cols-[minmax(0,1fr)_44px_44px_44px] gap-2 border-b border-gray-900/80 px-2 py-1.5 text-left text-[10px] font-mono transition-colors last:border-b-0 {scopeMode === 'benchmark' && selectedBenchmarkGroup === row.key ? 'bg-amber-950/40 text-amber-200' : 'text-gray-400 hover:bg-gray-900/60 hover:text-gray-200'}"
							title={`Latest run #${row.latestId} · ${row.groupCount} originals rerun · ${row.errorCount} errors`}
						>
							<span class="truncate">{row.label}</span>
							<span class="text-right tabular-nums">{row.runCount}</span>
							<span class="text-right tabular-nums">{formatMetric(row.bestValBpb)}</span>
							<span class="text-right tabular-nums">{formatMetric(row.avgValBpb)}</span>
						</button>
					{/each}
				</div>
			</div>
		{:else}
			<p class="text-[10px] font-mono text-gray-600">No benchmark rerun cohorts yet.</p>
		{/if}
	</div>

	<div class="space-y-1">
		<div class="flex items-center justify-between">
			<p class="text-[10px] font-mono uppercase tracking-[0.14em] text-gray-500">model families</p>
			<span class="text-[10px] font-mono text-gray-600">{familyRows.length} families</span>
		</div>
		<div class="rounded border border-gray-800 overflow-hidden">
			<div class="grid grid-cols-[minmax(0,1fr)_44px_44px_44px] gap-2 border-b border-gray-800 bg-gray-950/80 px-2 py-1 text-[9px] font-mono uppercase tracking-[0.14em] text-gray-500">
				<span>family</span>
				<span class="text-right">runs</span>
				<span class="text-right">{metricShortLabel}</span>
				<span class="text-right">sets</span>
			</div>
			<div class="max-h-36 overflow-y-auto">
				{#each familyRows as row}
					<button
						type="button"
						onclick={() => onSelectModelFamily?.(row.key)}
						class="grid w-full grid-cols-[minmax(0,1fr)_44px_44px_44px] gap-2 border-b border-gray-900/80 px-2 py-1.5 text-left text-[10px] font-mono transition-colors last:border-b-0 {scopeMode === 'family' && selectedModelFamily === row.key ? 'bg-blue-950/50 text-blue-200' : 'text-gray-400 hover:bg-gray-900/60 hover:text-gray-200'}"
						title={`${row.errorCount} errors · ${row.groupCount} benchmark groups`}
					>
						<span class="truncate">{row.label}</span>
						<span class="text-right tabular-nums">{row.runCount}</span>
						<span class="text-right tabular-nums">{formatMetric(row.bestValBpb)}</span>
						<span class="text-right tabular-nums">{row.groupCount}</span>
					</button>
				{/each}
			</div>
		</div>
	</div>
</div>
