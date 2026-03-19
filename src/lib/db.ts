import type { ExperimentRecord } from './research/prompt';
import type { ExperimentEvalSummary } from './research/metrics';

export type ExperimentRow = {
	id: number;
	project_id: number | null;
	name: string;
	source: 'manual' | 'auto';
	code: string;
	dataset_version_id: number | null;
	dataset_label: string | null;
	dataset_source_ref: string | null;
	trainer_key: string;
	model_family: string;
	reasoning: string;
	val_bpb: number;
	primary_score: number | null;
	elapsed: number;
	total_steps: number;
	kept: number;
	error: string | null;
	loss_curve: string | null;
	eval_summary_json: string | null;
	weights_path: string | null;
	rerun_of: number | null;
	benchmark_group: string | null;
	created_at: string;
};

export type InferenceRow = {
	id: number;
	experiment_id: number;
	prompt: string;
	output: string;
	temperature: number;
	created_at: string;
};

type ImportSummary = {
	addedWorkspaces: number;
	skippedWorkspaces: number;
	addedExperiments: number;
	skippedExperiments: number;
	addedLossSteps: number;
	skippedLossSteps: number;
	addedInferences: number;
	skippedInferences: number;
	addedWeights: number;
	skippedWeights: number;
};

const INVALID_SCORE_SENTINEL = 1e308;

async function readJson<T>(response: Response): Promise<T> {
	if (!response.ok) {
		const message = await response.text();
		throw new Error(message || `HTTP ${response.status}`);
	}
	return response.json() as Promise<T>;
}

function encodeScore(value: number): number {
	return Number.isFinite(value) ? value : INVALID_SCORE_SENTINEL;
}

function decodeScore(value: unknown): number {
	if (value == null) return Number.POSITIVE_INFINITY;
	if (typeof value !== 'number') return Number(value);
	return value >= INVALID_SCORE_SENTINEL ? Number.POSITIVE_INFINITY : value;
}

function decodeExperimentRecord(exp: ExperimentRecord): ExperimentRecord {
	return {
		...exp,
		valBpb: decodeScore(exp.valBpb),
		primaryScore: exp.primaryScore == null ? null : decodeScore(exp.primaryScore)
	};
}

function decodeExperimentRow(row: ExperimentRow | null): ExperimentRow | null {
	if (!row) return null;
	return {
		...row,
		val_bpb: decodeScore(row.val_bpb),
		primary_score: row.primary_score == null ? null : decodeScore(row.primary_score)
	};
}

export async function getDb(): Promise<void> {
	await readJson<{ experiments: ExperimentRecord[] }>(await fetch('/api/experiments'));
}

export async function insertExperiment(exp: {
	projectId?: number | null;
	name?: string;
	source?: 'manual' | 'auto';
	code: string;
	datasetVersionId?: number | null;
	datasetLabel?: string | null;
	datasetSourceRef?: string | null;
	trainerKey?: string;
	modelFamily?: string;
	valBpb: number;
	primaryScore?: number | null;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	lossCurve?: { step: number; loss: number }[];
	error?: string;
	evalSummary?: ExperimentEvalSummary | null;
	rerunOf?: number | null;
	benchmarkGroup?: string | null;
	createdAt?: string;
}): Promise<number> {
	const data = await readJson<{ id: number }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'insertExperiment',
			experiment: {
				...exp,
				valBpb: encodeScore(exp.valBpb),
				primaryScore: exp.primaryScore == null ? null : encodeScore(exp.primaryScore)
			}
		})
	}));
	return data.id;
}

export async function getAllExperimentRecords(datasetVersionId?: number | null, projectId?: number | null): Promise<ExperimentRecord[]> {
	const url = new URL('/api/experiments', window.location.origin);
	if (datasetVersionId != null) {
		url.searchParams.set('datasetVersionId', String(datasetVersionId));
	}
	if (projectId != null) {
		url.searchParams.set('projectId', String(projectId));
	}
	const data = await readJson<{ experiments: ExperimentRecord[] }>(await fetch(url));
	return data.experiments.map(decodeExperimentRecord);
}

export async function getBestExperiment(datasetVersionId?: number | null): Promise<ExperimentRow | null> {
	const url = new URL('/api/experiments/best', window.location.origin);
	if (datasetVersionId != null) {
		url.searchParams.set('datasetVersionId', String(datasetVersionId));
	}
	const data = await readJson<{ best: ExperimentRow | null }>(await fetch(url));
	return decodeExperimentRow(data.best);
}

export async function updateWeightsPath(id: number, weightsPath: string): Promise<void> {
	await readJson<{ ok: true }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'updateWeightsPath',
			id,
			weightsPath
		})
	}));
}

export async function deleteExperiments(ids: number[]): Promise<number> {
	if (ids.length === 0) return 0;
	const data = await readJson<{ deleted: number }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'deleteExperiments',
			ids
		})
	}));
	return data.deleted;
}

export async function clearAllData(): Promise<void> {
	await readJson<{ ok: true }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'clearAllData'
		})
	}));
}

export async function insertLossCurve(experimentId: number, curve: { step: number; loss: number }[]): Promise<void> {
	if (curve.length === 0) return;
	await readJson<{ ok: true }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'insertLossCurve',
			experimentId,
			curve
		})
	}));
}

export async function insertInference(inf: {
	experimentId: number;
	prompt: string;
	output: string;
	temperature: number;
	createdAt?: string;
}): Promise<number> {
	const data = await readJson<{ id: number }>(await fetch('/api/experiments', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'insertInference',
			inference: inf
		})
	}));
	return data.id;
}

export async function getInferencesForExperiment(experimentId: number): Promise<InferenceRow[]> {
	const data = await readJson<{ inferences: InferenceRow[] }>(await fetch(`/api/experiments/${experimentId}/inferences`));
	return data.inferences;
}

export async function exportCsvZip(projectId?: number | null): Promise<Blob> {
	const url = new URL('/api/experiments/export', window.location.origin);
	if (projectId != null) {
		url.searchParams.set('projectId', String(projectId));
	}
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(await response.text());
	}
	return response.blob();
}

export async function importCsvZip(file: Blob): Promise<ImportSummary> {
	const formData = new FormData();
	formData.set('file', file, 'autoresearch-experiments.zip');
	return readJson<ImportSummary>(await fetch('/api/experiments/import', {
		method: 'POST',
		body: formData
	}));
}
