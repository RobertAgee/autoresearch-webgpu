import JSZip from 'jszip';
import type { ExperimentRecord } from '$lib/research/prompt';
import type { ExperimentEvalSummary } from '$lib/research/metrics';
import { getCatalogDb } from './catalog-db';
import { DEFAULT_TRAINER_KEY } from '$lib/trainers';
import {
	clearWeightsArtifacts,
	deleteWeightsArtifact,
	loadWeightsArtifact,
	saveWeightsArtifact,
	type WeightMeta
} from './artifacts-service';
import {
	insertExperimentWorkspaceRow,
	listExperimentWorkspaceRows,
	updateExperimentWorkspaceBaseExperimentRow,
	type ExperimentWorkspaceRow
} from './experiment-workspaces-service';

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

export type ImportSummary = {
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

type CsvRow = Record<string, string>;

type ImportedExperiment = {
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
	kept: boolean;
	error: string | null;
	loss_curve: { step: number; loss: number }[] | null;
	eval_summary_json: string | null;
	rerun_of: number | null;
	benchmark_group: string | null;
	created_at: string;
};

type ImportedWorkspace = {
	id: number;
	name: string;
	source_type: 'dataset' | 'results';
	dataset_version_id: number | null;
	base_experiment_id: number | null;
	readme: string;
	notes: string;
	created_at: string;
};

type ImportedLossStep = {
	experiment_id: number;
	step: number;
	loss: number;
};

type ImportedInference = {
	experiment_id: number;
	prompt: string;
	output: string;
	temperature: number;
	created_at: string;
};

const INVALID_SCORE_SENTINEL = 1e308;

function parseLossCurve(value: string | null): { step: number; loss: number }[] | undefined {
	if (!value) return undefined;
	try {
		return JSON.parse(value) as { step: number; loss: number }[];
	} catch {
		return undefined;
	}
}

function parseEvalSummary(value: string | null): ExperimentEvalSummary | null {
	if (!value) return null;
	try {
		return JSON.parse(value) as ExperimentEvalSummary;
	} catch {
		return null;
	}
}

function normalizeStoredScore(value: unknown): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		return INVALID_SCORE_SENTINEL;
	}
	return value;
}

function decodeStoredScore(value: number): number {
	return value >= INVALID_SCORE_SENTINEL ? Number.POSITIVE_INFINITY : value;
}

function rowToRecord(row: ExperimentRow): ExperimentRecord {
	return {
		id: row.id,
		projectId: row.project_id,
		name: row.name,
		source: row.source,
		code: row.code,
		datasetVersionId: row.dataset_version_id,
		datasetLabel: row.dataset_label,
		datasetSourceRef: row.dataset_source_ref,
		trainerKey: row.trainer_key,
		modelFamily: row.model_family,
		valBpb: decodeStoredScore(row.val_bpb),
		primaryScore: row.primary_score == null ? null : decodeStoredScore(row.primary_score),
		elapsed: row.elapsed,
		totalSteps: row.total_steps,
		reasoning: row.reasoning,
		kept: row.kept === 1,
		error: row.error ?? undefined,
		lossCurve: parseLossCurve(row.loss_curve),
		evalSummary: parseEvalSummary(row.eval_summary_json),
		rerunOf: row.rerun_of,
		benchmarkGroup: row.benchmark_group,
		createdAt: row.created_at
	};
}

export function ensureExperimentStorage(): void {
	getCatalogDb();
}

export function insertExperimentRow(exp: {
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
}): number {
	const db = getCatalogDb();
	const result = db.prepare(`
		INSERT INTO experiments (
			project_id,
			name,
			source,
			code,
			dataset_version_id,
			dataset_label,
			dataset_source_ref,
			trainer_key,
			model_family,
			val_bpb,
			primary_score,
			elapsed,
			total_steps,
			reasoning,
			kept,
			loss_curve,
			error,
			eval_summary_json,
			rerun_of,
			benchmark_group,
			created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
	`).run(
		exp.projectId ?? null,
		exp.name ?? '',
		exp.source ?? 'manual',
		exp.code,
		exp.datasetVersionId ?? null,
		exp.datasetLabel ?? null,
		exp.datasetSourceRef ?? null,
		exp.trainerKey ?? DEFAULT_TRAINER_KEY,
		exp.modelFamily ?? 'byte-gpt',
		normalizeStoredScore(exp.valBpb),
		exp.primaryScore == null ? null : normalizeStoredScore(exp.primaryScore),
		exp.elapsed,
		exp.totalSteps,
		exp.reasoning,
		exp.kept ? 1 : 0,
		exp.lossCurve ? JSON.stringify(exp.lossCurve) : null,
		exp.error ?? null,
		exp.evalSummary ? JSON.stringify(exp.evalSummary) : null,
		exp.rerunOf ?? null,
		exp.benchmarkGroup ?? null,
		exp.createdAt ?? null
	);
	return Number(result.lastInsertRowid);
}

export function insertLossCurveRows(experimentId: number, curve: { step: number; loss: number }[]): void {
	if (curve.length === 0) return;
	const db = getCatalogDb();
	const stmt = db.prepare(`
		INSERT INTO loss_steps (experiment_id, step, loss)
		VALUES (?, ?, ?)
	`);
	for (const point of curve) {
		stmt.run(experimentId, point.step, point.loss);
	}
}

export function insertInferenceRow(inf: {
	experimentId: number;
	prompt: string;
	output: string;
	temperature: number;
	createdAt?: string;
}): number {
	const db = getCatalogDb();
	const result = db.prepare(`
		INSERT INTO inferences (experiment_id, prompt, output, temperature, created_at)
		VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
	`).run(
		inf.experimentId,
		inf.prompt,
		inf.output,
		inf.temperature,
		inf.createdAt ?? null
	);
	return Number(result.lastInsertRowid);
}

export function updateExperimentWeightsPath(id: number, weightsPath: string): void {
	getCatalogDb().prepare(`
		UPDATE experiments
		SET weights_path = ?
		WHERE id = ?
	`).run(weightsPath, id);
}

export function listExperimentRows(datasetVersionId?: number | null, projectId?: number | null): ExperimentRow[] {
	const db = getCatalogDb();
	if (projectId != null) {
		return db.prepare(`SELECT * FROM experiments WHERE project_id = ? ORDER BY id`).all(projectId) as ExperimentRow[];
	}
	return datasetVersionId == null
		? db.prepare(`SELECT * FROM experiments ORDER BY id`).all() as ExperimentRow[]
		: db.prepare(`SELECT * FROM experiments WHERE dataset_version_id = ? ORDER BY id`).all(datasetVersionId) as ExperimentRow[];
}

export function listExperimentRecords(datasetVersionId?: number | null, projectId?: number | null): ExperimentRecord[] {
	const rows = listExperimentRows(datasetVersionId, projectId);
	const db = getCatalogDb();
	const lossRows = db.prepare(`
		SELECT experiment_id, step, loss
		FROM loss_steps
		ORDER BY experiment_id, step
	`).all() as { experiment_id: number; step: number; loss: number }[];
	const curves = new Map<number, { step: number; loss: number }[]>();
	for (const row of lossRows) {
		const existing = curves.get(row.experiment_id) ?? [];
		existing.push({ step: row.step, loss: row.loss });
		curves.set(row.experiment_id, existing);
	}
	return rows.map((row) => ({
		...rowToRecord(row),
		lossCurve: curves.get(row.id) ?? parseLossCurve(row.loss_curve)
	}));
}

export function getBestExperimentRow(datasetVersionId?: number | null): ExperimentRow | null {
	const db = getCatalogDb();
	const row = datasetVersionId == null
		? db.prepare(`
			SELECT *
			FROM experiments
			WHERE error IS NULL
			  AND kept = 1
			  AND primary_score IS NOT NULL
			ORDER BY primary_score DESC, created_at DESC, id DESC
			LIMIT 1
		`).get()
		: db.prepare(`
			SELECT *
			FROM experiments
			WHERE dataset_version_id = ?
			  AND error IS NULL
			  AND kept = 1
			  AND primary_score IS NOT NULL
			ORDER BY primary_score DESC, created_at DESC, id DESC
			LIMIT 1
		`).get(datasetVersionId);
	return row as ExperimentRow | undefined ?? null;
}

export function listInferencesForExperiment(experimentId: number): InferenceRow[] {
	return getCatalogDb().prepare(`
		SELECT *
		FROM inferences
		WHERE experiment_id = ?
		ORDER BY created_at DESC
	`).all(experimentId) as InferenceRow[];
}

export async function deleteExperimentRows(ids: number[]): Promise<number> {
	if (ids.length === 0) return 0;
	const db = getCatalogDb();
	const stmt = db.prepare(`DELETE FROM experiments WHERE id = ?`);
	let deleted = 0;
	for (const id of ids) {
		await deleteWeightsArtifact(id);
		const result = stmt.run(id);
		deleted += result.changes;
	}
	return deleted;
}

export async function clearExperimentData(): Promise<void> {
	const db = getCatalogDb();
	db.exec(`
		DELETE FROM inferences;
		DELETE FROM loss_steps;
		DELETE FROM experiments;
	`);
	await clearWeightsArtifacts();
}

function toCsv(rows: Record<string, unknown>[]): string {
	if (rows.length === 0) return '';
	const keys = Object.keys(rows[0]);
	const escape = (value: unknown) => {
		const raw = typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
		return raw.includes(',') || raw.includes('"') || raw.includes('\n')
			? `"${raw.replace(/"/g, '""')}"`
			: raw;
	};
	return [keys.join(','), ...rows.map((row) => keys.map((key) => escape(row[key])).join(','))].join('\n');
}

export async function exportExperimentsZip(): Promise<Uint8Array> {
	const db = getCatalogDb();
	const zip = new JSZip();
	const experiments = db.prepare(`SELECT * FROM experiments ORDER BY id`).all() as ExperimentRow[];
	const workspaces = listExperimentWorkspaceRows();
	zip.file('experiment_workspaces.csv', toCsv(workspaces as unknown as Record<string, unknown>[]));
	zip.file('experiments.csv', toCsv(experiments as unknown as Record<string, unknown>[]));
	zip.file('loss_steps.csv', toCsv(db.prepare(`SELECT * FROM loss_steps ORDER BY experiment_id, step`).all() as Record<string, unknown>[]));
	zip.file('inferences.csv', toCsv(db.prepare(`SELECT * FROM inferences ORDER BY experiment_id, created_at`).all() as Record<string, unknown>[]));
	for (const experiment of experiments) {
		if (!experiment.weights_path) continue;
		const artifact = await loadWeightsArtifact(experiment.id);
		if (!artifact) continue;
		zip.file(`weights/exp-${experiment.id}.bin`, artifact.buffer);
		zip.file(`weights/exp-${experiment.id}.meta.json`, JSON.stringify(artifact.metas));
	}
	return zip.generateAsync({ type: 'uint8array' });
}

export async function exportProjectExperimentsZip(projectId: number): Promise<Uint8Array> {
	const db = getCatalogDb();
	const zip = new JSZip();
	const workspaces = db.prepare(`
		SELECT *
		FROM experiment_workspaces
		WHERE id = ?
		ORDER BY id
	`).all(projectId) as ExperimentWorkspaceRow[];
	const experiments = db.prepare(`SELECT * FROM experiments WHERE project_id = ? ORDER BY id`).all(projectId) as ExperimentRow[];
	zip.file('experiment_workspaces.csv', toCsv(workspaces as unknown as Record<string, unknown>[]));
	zip.file('experiments.csv', toCsv(experiments as unknown as Record<string, unknown>[]));
	zip.file(
		'loss_steps.csv',
		toCsv(
			db.prepare(`
				SELECT *
				FROM loss_steps
				WHERE experiment_id IN (SELECT id FROM experiments WHERE project_id = ?)
				ORDER BY experiment_id, step
			`).all(projectId) as Record<string, unknown>[]
		)
	);
	zip.file(
		'inferences.csv',
		toCsv(
			db.prepare(`
				SELECT *
				FROM inferences
				WHERE experiment_id IN (SELECT id FROM experiments WHERE project_id = ?)
				ORDER BY experiment_id, created_at
			`).all(projectId) as Record<string, unknown>[]
		)
	);
	for (const experiment of experiments) {
		if (!experiment.weights_path) continue;
		const artifact = await loadWeightsArtifact(experiment.id);
		if (!artifact) continue;
		zip.file(`weights/exp-${experiment.id}.bin`, artifact.buffer);
		zip.file(`weights/exp-${experiment.id}.meta.json`, JSON.stringify(artifact.metas));
	}
	return zip.generateAsync({ type: 'uint8array' });
}

function parseCsv(text: string): CsvRow[] {
	if (!text.trim()) return [];
	const rows: string[][] = [];
	let currentRow: string[] = [];
	let currentField = '';
	let inQuotes = false;

	for (let i = 0; i < text.length; i++) {
		const char = text[i];
		if (inQuotes) {
			if (char === '"') {
				if (text[i + 1] === '"') {
					currentField += '"';
					i++;
				} else {
					inQuotes = false;
				}
			} else {
				currentField += char;
			}
			continue;
		}
		if (char === '"') {
			inQuotes = true;
		} else if (char === ',') {
			currentRow.push(currentField);
			currentField = '';
		} else if (char === '\n') {
			currentRow.push(currentField);
			rows.push(currentRow);
			currentRow = [];
			currentField = '';
		} else if (char !== '\r') {
			currentField += char;
		}
	}
	currentRow.push(currentField);
	if (currentRow.length > 1 || currentRow[0] !== '') {
		rows.push(currentRow);
	}
	const [header, ...dataRows] = rows;
	if (!header) return [];
	return dataRows
		.filter((row) => row.some((value) => value !== ''))
		.map((row) => Object.fromEntries(header.map((key, index) => [key, row[index] ?? ''])));
}

function parseNumber(value: string, field: string): number {
	const parsed = Number(value);
	if (Number.isNaN(parsed)) throw new Error(`Invalid number for ${field}: ${value}`);
	return parsed;
}

function parseNullableNumber(value: string | undefined, field: string): number | null {
	if (value == null || value === '' || value === 'null') return null;
	return parseNumber(value, field);
}

function parseBoolean(value: string): boolean {
	return value === 'true' || value === 't' || value === '1';
}

function parseJsonField<T>(value: string, fallback: T): T {
	if (!value) return fallback;
	try {
		return JSON.parse(value) as T;
	} catch {
		return fallback;
	}
}

function experimentKey(exp: Pick<ExperimentRow, 'name' | 'source' | 'code' | 'dataset_version_id' | 'dataset_label' | 'dataset_source_ref' | 'trainer_key' | 'model_family' | 'reasoning' | 'val_bpb' | 'primary_score' | 'elapsed' | 'total_steps' | 'kept' | 'error' | 'eval_summary_json' | 'rerun_of' | 'benchmark_group' | 'created_at'>): string {
	return JSON.stringify([
		exp.name,
		exp.source,
		exp.code,
		exp.dataset_version_id ?? null,
		exp.dataset_label ?? null,
		exp.dataset_source_ref ?? null,
		exp.trainer_key,
		exp.model_family,
		exp.reasoning,
		exp.val_bpb,
		exp.primary_score ?? null,
		exp.elapsed,
		exp.total_steps,
		exp.kept,
		exp.error ?? null,
		exp.eval_summary_json ?? null,
		exp.rerun_of ?? null,
		exp.benchmark_group ?? null,
		exp.created_at
	]);
}

function importedExperimentKey(exp: ImportedExperiment, idMap: Map<number, number>): string {
	return JSON.stringify([
		exp.name,
		exp.source,
		exp.code,
		exp.dataset_version_id ?? null,
		exp.dataset_label ?? null,
		exp.dataset_source_ref ?? null,
		exp.trainer_key,
		exp.model_family,
		exp.reasoning,
		exp.val_bpb,
		exp.primary_score ?? null,
		exp.elapsed,
		exp.total_steps,
		exp.kept ? 1 : 0,
		exp.error ?? null,
		exp.eval_summary_json ?? null,
		exp.rerun_of != null ? idMap.get(exp.rerun_of) ?? null : null,
		exp.benchmark_group ?? null,
		exp.created_at
	]);
}

function lossStepKey(experimentId: number, step: number, loss: number): string {
	return `${experimentId}:${step}:${loss}`;
}

function inferenceKey(inf: Pick<InferenceRow, 'experiment_id' | 'prompt' | 'output' | 'temperature' | 'created_at'>): string {
	return JSON.stringify([inf.experiment_id, inf.prompt, inf.output, inf.temperature, inf.created_at]);
}

function parseImportedExperiments(rows: CsvRow[]): ImportedExperiment[] {
	return rows.map((row) => ({
		id: parseNumber(row.id, 'experiments.id'),
		project_id: parseNullableNumber(row.project_id, 'experiments.project_id'),
		name: row.name ?? '',
		source: row.source === 'auto' ? 'auto' : 'manual',
		code: row.code ?? '',
		dataset_version_id: parseNullableNumber(row.dataset_version_id, 'experiments.dataset_version_id'),
		dataset_label: row.dataset_label || null,
		dataset_source_ref: row.dataset_source_ref || null,
		trainer_key: row.trainer_key || DEFAULT_TRAINER_KEY,
		model_family: row.model_family || 'byte-gpt',
		reasoning: row.reasoning ?? '',
		val_bpb: parseNumber(row.val_bpb, 'experiments.val_bpb'),
		primary_score: parseNullableNumber(row.primary_score, 'experiments.primary_score'),
		elapsed: parseNumber(row.elapsed, 'experiments.elapsed'),
		total_steps: parseNumber(row.total_steps, 'experiments.total_steps'),
		kept: parseBoolean(row.kept ?? ''),
		error: row.error ? row.error : null,
		loss_curve: parseJsonField(row.loss_curve ?? '', null),
		eval_summary_json: row.eval_summary_json || null,
		rerun_of: parseNullableNumber(row.rerun_of, 'experiments.rerun_of'),
		benchmark_group: row.benchmark_group || null,
		created_at: row.created_at || new Date().toISOString()
	}));
}

function parseImportedWorkspaces(rows: CsvRow[]): ImportedWorkspace[] {
	return rows.map((row) => ({
		id: parseNumber(row.id, 'experiment_workspaces.id'),
		name: row.name ?? '',
		source_type: row.source_type === 'results' ? 'results' : 'dataset',
		dataset_version_id: parseNullableNumber(row.dataset_version_id, 'experiment_workspaces.dataset_version_id'),
		base_experiment_id: parseNullableNumber(row.base_experiment_id, 'experiment_workspaces.base_experiment_id'),
		readme: row.readme ?? '',
		notes: row.notes ?? '',
		created_at: row.created_at || new Date().toISOString()
	}));
}

function workspaceKey(workspace: Pick<ExperimentWorkspaceRow, 'name' | 'source_type' | 'dataset_version_id' | 'readme' | 'notes' | 'created_at'>): string {
	return JSON.stringify([
		workspace.name,
		workspace.source_type,
		workspace.dataset_version_id ?? null,
		workspace.readme,
		workspace.notes,
		workspace.created_at
	]);
}

function importedWorkspaceKey(workspace: ImportedWorkspace): string {
	return JSON.stringify([
		workspace.name,
		workspace.source_type,
		workspace.dataset_version_id ?? null,
		workspace.readme,
		workspace.notes,
		workspace.created_at
	]);
}

function parseImportedLossSteps(rows: CsvRow[]): ImportedLossStep[] {
	return rows.map((row) => ({
		experiment_id: parseNumber(row.experiment_id, 'loss_steps.experiment_id'),
		step: parseNumber(row.step, 'loss_steps.step'),
		loss: parseNumber(row.loss, 'loss_steps.loss')
	}));
}

function parseImportedInferences(rows: CsvRow[]): ImportedInference[] {
	return rows.map((row) => ({
		experiment_id: parseNumber(row.experiment_id, 'inferences.experiment_id'),
		prompt: row.prompt ?? '',
		output: row.output ?? '',
		temperature: parseNumber(row.temperature, 'inferences.temperature'),
		created_at: row.created_at || new Date().toISOString()
	}));
}

export async function importExperimentsZip(file: Blob): Promise<ImportSummary> {
	const zip = await JSZip.loadAsync(await file.arrayBuffer());
	const experimentsFile = zip.file('experiments.csv');
	if (!experimentsFile) throw new Error('Import ZIP is missing experiments.csv');

	const [workspacesCsv, experimentsCsv, lossStepsCsv, inferencesCsv] = await Promise.all([
		zip.file('experiment_workspaces.csv')?.async('string') ?? Promise.resolve(''),
		experimentsFile.async('string'),
		zip.file('loss_steps.csv')?.async('string') ?? Promise.resolve(''),
		zip.file('inferences.csv')?.async('string') ?? Promise.resolve('')
	]);

	const importedWorkspaces = parseImportedWorkspaces(parseCsv(workspacesCsv)).sort((a, b) => a.id - b.id);
	const importedExperiments = parseImportedExperiments(parseCsv(experimentsCsv)).sort((a, b) => a.id - b.id);
	const importedLossSteps = parseImportedLossSteps(parseCsv(lossStepsCsv));
	const importedInferences = parseImportedInferences(parseCsv(inferencesCsv));

	const summary: ImportSummary = {
		addedWorkspaces: 0,
		skippedWorkspaces: 0,
		addedExperiments: 0,
		skippedExperiments: 0,
		addedLossSteps: 0,
		skippedLossSteps: 0,
		addedInferences: 0,
		skippedInferences: 0,
		addedWeights: 0,
		skippedWeights: 0
	};

	const db = getCatalogDb();
	const existingWorkspaces = listExperimentWorkspaceRows();
	const existingWorkspaceKeys = new Map(existingWorkspaces.map((row) => [workspaceKey(row), row.id]));
	const existingExperiments = listExperimentRows();
	const existingExperimentKeys = new Map(existingExperiments.map((row) => [experimentKey(row), row.id]));
	const existingLossStepKeys = new Set(
		(db.prepare(`SELECT experiment_id, step, loss FROM loss_steps`).all() as { experiment_id: number; step: number; loss: number }[])
			.map((row) => lossStepKey(row.experiment_id, row.step, row.loss))
	);
	const existingInferenceKeys = new Set(
		(db.prepare(`SELECT experiment_id, prompt, output, temperature, created_at FROM inferences`).all() as InferenceRow[])
			.map(inferenceKey)
	);
	const workspaceIdMap = new Map<number, number>();
	const idMap = new Map<number, number>();
	const savedWeightExperimentIds: number[] = [];

	db.exec('BEGIN');
	try {
		for (const workspace of importedWorkspaces) {
			const key = importedWorkspaceKey(workspace);
			const existingId = existingWorkspaceKeys.get(key);
			if (existingId != null) {
				workspaceIdMap.set(workspace.id, existingId);
				summary.skippedWorkspaces++;
				continue;
			}

			const insertedId = insertExperimentWorkspaceRow({
				name: workspace.name,
				sourceType: workspace.source_type,
				datasetVersionId: workspace.dataset_version_id,
				baseExperimentId: null,
				readme: workspace.readme,
				notes: workspace.notes,
				createdAt: workspace.created_at
			});
			workspaceIdMap.set(workspace.id, insertedId);
			existingWorkspaceKeys.set(key, insertedId);
			summary.addedWorkspaces++;
		}

		for (const exp of importedExperiments) {
			const key = importedExperimentKey(exp, idMap);
			const existingId = existingExperimentKeys.get(key);
			if (existingId != null) {
				idMap.set(exp.id, existingId);
				summary.skippedExperiments++;
				continue;
			}

			const insertedId = insertExperimentRow({
				projectId: exp.project_id != null ? workspaceIdMap.get(exp.project_id) ?? null : null,
				name: exp.name,
				source: exp.source,
				code: exp.code,
				datasetVersionId: exp.dataset_version_id,
				datasetLabel: exp.dataset_label,
				datasetSourceRef: exp.dataset_source_ref,
				trainerKey: exp.trainer_key,
				modelFamily: exp.model_family,
				valBpb: exp.val_bpb,
				primaryScore: exp.primary_score,
				elapsed: exp.elapsed,
				totalSteps: exp.total_steps,
				reasoning: exp.reasoning,
				kept: exp.kept,
				lossCurve: exp.loss_curve ?? undefined,
				error: exp.error ?? undefined,
				evalSummary: parseEvalSummary(exp.eval_summary_json),
				rerunOf: null,
				benchmarkGroup: exp.benchmark_group,
				createdAt: exp.created_at
			});
			idMap.set(exp.id, insertedId);
			existingExperimentKeys.set(importedExperimentKey(exp, idMap), insertedId);
			summary.addedExperiments++;
		}

		for (const workspace of importedWorkspaces) {
			const mappedWorkspaceId = workspaceIdMap.get(workspace.id);
			if (mappedWorkspaceId == null) continue;
			updateExperimentWorkspaceBaseExperimentRow(
				mappedWorkspaceId,
				workspace.base_experiment_id != null ? idMap.get(workspace.base_experiment_id) ?? null : null
			);
		}

		const updateRerunStmt = db.prepare(`
			UPDATE experiments
			SET rerun_of = ?, benchmark_group = ?
			WHERE id = ?
		`);
		for (const exp of importedExperiments) {
			const id = idMap.get(exp.id);
			if (id == null) continue;
			updateRerunStmt.run(
				exp.rerun_of != null ? idMap.get(exp.rerun_of) ?? null : null,
				exp.benchmark_group ?? null,
				id
			);
		}

		for (const row of importedLossSteps) {
			const experimentId = idMap.get(row.experiment_id);
			if (experimentId == null) continue;
			const key = lossStepKey(experimentId, row.step, row.loss);
			if (existingLossStepKeys.has(key)) {
				summary.skippedLossSteps++;
				continue;
			}
			db.prepare(`
				INSERT INTO loss_steps (experiment_id, step, loss)
				VALUES (?, ?, ?)
			`).run(experimentId, row.step, row.loss);
			existingLossStepKeys.add(key);
			summary.addedLossSteps++;
		}

		for (const row of importedInferences) {
			const experimentId = idMap.get(row.experiment_id);
			if (experimentId == null) continue;
			const key = inferenceKey({
				id: 0,
				experiment_id: experimentId,
				prompt: row.prompt,
				output: row.output,
				temperature: row.temperature,
				created_at: row.created_at
			});
			if (existingInferenceKeys.has(key)) {
				summary.skippedInferences++;
				continue;
			}
			insertInferenceRow({
				experimentId,
				prompt: row.prompt,
				output: row.output,
				temperature: row.temperature,
				createdAt: row.created_at
			});
			existingInferenceKeys.add(key);
			summary.addedInferences++;
		}

		const updateWeightsStmt = db.prepare(`
			UPDATE experiments
			SET weights_path = ?
			WHERE id = ?
		`);
		for (const exp of importedExperiments) {
			const mappedExperimentId = idMap.get(exp.id);
			if (mappedExperimentId == null) continue;

			const weightDataFile = zip.file(`weights/exp-${exp.id}.bin`);
			const weightMetaFile = zip.file(`weights/exp-${exp.id}.meta.json`);
			if (!weightDataFile || !weightMetaFile) continue;

			const existingArtifact = await loadWeightsArtifact(mappedExperimentId);
			if (existingArtifact) {
				summary.skippedWeights++;
				continue;
			}

			const [weightBuffer, weightMetaText] = await Promise.all([
				weightDataFile.async('uint8array'),
				weightMetaFile.async('string')
			]);
			const savedPath = await saveWeightsArtifact(
				mappedExperimentId,
				weightBuffer,
				JSON.parse(weightMetaText) as WeightMeta[]
			);
			updateWeightsStmt.run(savedPath, mappedExperimentId);
			savedWeightExperimentIds.push(mappedExperimentId);
			summary.addedWeights++;
		}

		db.exec('COMMIT');
		return summary;
	} catch (error) {
		db.exec('ROLLBACK');
		await Promise.all(savedWeightExperimentIds.map((id) => deleteWeightsArtifact(id)));
		throw error;
	}
}
