import type { EvalReport } from './eval';
import type { ExperimentRecord, ResearchDatasetContext } from './prompt';

export type MetricDirection = 'higher' | 'lower';

export type ExperimentEvalSummary = {
	evaluatorKey: string;
	label: string;
	stage: string;
	phase: string;
	gated: boolean;
	sampleCount: number;
	primaryMetricKey: string;
	primaryMetricLabel: string;
	primaryMetricDirection: MetricDirection;
	primaryScore: number;
	metrics: Record<string, number | boolean | null>;
};

export type ExperimentMetricColumn = {
	key: string;
	label: string;
	title: string;
	width: string;
	value: (experiment: ExperimentRecord) => number | boolean | null;
	render: (value: number | boolean | null) => string;
};

export const ABC_EVALUATOR_KEY = 'abc-structure-v1';

function asRate(value: number | boolean | null): string {
	return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '--';
}

function asScore(value: number | boolean | null): string {
	return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '--';
}

function asGate(value: number | boolean | null): string {
	return value === true ? 'pass' : value === false ? 'fail' : '--';
}

export function buildEvalSummary(
	report: EvalReport | null,
	stage: string,
	phase: string
): ExperimentEvalSummary | null {
	if (!report) return null;

	return {
		evaluatorKey: ABC_EVALUATOR_KEY,
		label: report.label,
		stage,
		phase,
		gated: report.gated,
		sampleCount: report.sampleCount,
		primaryMetricKey: 'compositeScore',
		primaryMetricLabel: 'abc score',
		primaryMetricDirection: 'higher',
		primaryScore: report.compositeScore,
		metrics: {
			structureScore: report.structureScore,
			headerRate: report.headerRate,
			kRate: report.kRate,
			lRate: report.lRate,
			terminationRate: report.terminationRate,
			promptFidelity: report.promptFidelity,
			repetitionScore: report.repetitionScore,
			gated: report.gated ? 1 : 0
		}
	};
}

export function getExperimentPrimaryMetric(experiment: ExperimentRecord): {
	label: string;
	direction: MetricDirection;
	value: number | null;
	shortLabel: string;
} {
	if (experiment.evalSummary && Number.isFinite(experiment.evalSummary.primaryScore)) {
		return {
			label: experiment.evalSummary.primaryMetricLabel,
			direction: experiment.evalSummary.primaryMetricDirection,
			value: experiment.evalSummary.primaryScore,
			shortLabel: experiment.evalSummary.primaryMetricLabel
		};
	}

	return {
		label: 'val bpb',
		direction: 'lower',
		value: Number.isFinite(experiment.valBpb) ? experiment.valBpb : null,
		shortLabel: 'bpb'
	};
}

export function getDatasetMetricColumns(context?: Pick<ResearchDatasetContext, 'recipeKey'> | null): ExperimentMetricColumn[] {
	if (!context?.recipeKey?.includes('abc')) {
		return [];
	}

	return [
		{
			key: 'gate',
			label: 'gate',
			title: 'Benchmark structure gate',
			width: '44px',
			value: (experiment) => experiment.evalSummary?.gated ?? null,
			render: asGate
		},
		{
			key: 'score',
			label: 'abc',
			title: 'Composite ABC benchmark score',
			width: '48px',
			value: (experiment) => experiment.evalSummary?.primaryScore ?? null,
			render: asScore
		},
		{
			key: 'end',
			label: 'end',
			title: 'Termination rate',
			width: '40px',
			value: (experiment) => experiment.evalSummary?.metrics.terminationRate ?? null,
			render: asRate
		},
		{
			key: 'prm',
			label: 'prm',
			title: 'Prompt fidelity score',
			width: '40px',
			value: (experiment) => experiment.evalSummary?.metrics.promptFidelity ?? null,
			render: asRate
		},
		{
			key: 'rep',
			label: 'rep',
			title: 'Repetition / anti-collapse score',
			width: '40px',
			value: (experiment) => experiment.evalSummary?.metrics.repetitionScore ?? null,
			render: asRate
		}
	];
}

export function formatExperimentMetricValue(experiment: ExperimentRecord): string {
	const primary = getExperimentPrimaryMetric(experiment);
	if (primary.value == null) return '--';
	return primary.direction === 'higher'
		? primary.value.toFixed(1)
		: primary.value.toFixed(3);
}
