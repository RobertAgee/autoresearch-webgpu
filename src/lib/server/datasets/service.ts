import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
	getDatasetRoot,
	getDatasetVersionRowById,
	getActiveDatasetVersionRow,
	insertDatasetVersion,
	listDatasetVersionRows,
	setActiveDatasetVersionRow,
	type DatasetVersionRow
} from '../catalog-db';
import { listApplicableDatasetRecipes, resolveDatasetRecipe } from './recipes';
import { DEFAULT_TRAINER_KEY, getTrainerDefinition } from '$lib/trainers';

type HuggingFaceSplitInfo = {
	name: string;
	num_examples: number;
};

type HuggingFaceInfoResponse = {
	dataset_info?: Record<string, {
		features?: Record<string, { dtype?: string }>;
		splits?: Record<string, HuggingFaceSplitInfo>;
		download_checksums?: Record<string, { num_bytes?: number }>;
	}>;
};

type HuggingFaceRowsResponse = {
	rows?: Array<{
		row?: Record<string, unknown>;
	}>;
};

type ImportRequest = {
	datasetId: string;
	label?: string;
	recipeKey?: string | null;
	textFields?: string[] | null;
	samplePrompt?: string | null;
	maxTrainExamples?: number | null;
	maxValidationExamples?: number | null;
};

type ImportSummary = {
	versions: ReturnType<typeof summarizeVersion>[];
	activeVersionId: number | null;
};

export type HuggingFaceDatasetInspection = {
	datasetId: string;
	configName: string;
	featureNames: string[];
	trainSplitName: string | null;
	trainExamples: number | null;
	validationSplitName: string | null;
	validationExamples: number | null;
	hasValidationSplit: boolean;
	splitNames: string[];
	trainSamples: Record<string, unknown>[];
	validationSamples: Record<string, unknown>[];
	recipeOptions: ReturnType<typeof listApplicableDatasetRecipes>;
	defaultProposal: DatasetRecipeProposal;
};

export type DatasetRecipeProposal = {
	datasetId: string;
	label: string;
	trainerKey: string;
	modelFamily: string;
	recipeKey: string;
	textFields: string[];
	maxTrainExamples: number | null;
	maxValidationExamples: number | null;
	validationStrategy: 'huggingface' | 'carve-from-train';
	preprocessingSummary: string;
	preprocessingSteps: string[];
	researchNotes: string[];
	samplePrompt: string;
	reasoning: string;
};

const LEGACY_SOURCE_REF = 'static/data';
const LEGACY_SLUG = 'legacy-static-data';
const LEGACY_LABEL = 'repo legacy dataset';
const LEGACY_RECIPE_KEY = 'legacy-byte-bin-v1';

function slugify(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/(^-|-$)/g, '') || 'dataset';
}

function parseHfPath(value: string): { sourceRef: string; revision: string; filePath: string } | null {
	const match = /^hf:\/\/datasets\/([^@]+)@([^/]+)\/(.+)$/.exec(value);
	if (!match) return null;
	return {
		sourceRef: match[1],
		revision: match[2],
		filePath: match[3]
	};
}

function checksumUrlToDownloadUrl(value: string): string {
	const parsed = parseHfPath(value);
	if (!parsed) {
		throw new Error(`Unsupported Hugging Face path: ${value}`);
	}
	return `https://huggingface.co/datasets/${parsed.sourceRef}/resolve/${parsed.revision}/${parsed.filePath}`;
}

async function fetchJson<T>(url: string): Promise<T> {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
	}
	return response.json() as Promise<T>;
}

async function fetchDatasetRows(url: string): Promise<Record<string, unknown>[]> {
	return fetchJson<Record<string, unknown>[]>(url);
}

async function fetchSplitPreviewRows(
	datasetId: string,
	configName: string,
	splitName: string | null,
	length = 3
): Promise<Record<string, unknown>[]> {
	if (!splitName) return [];
	try {
		const response = await fetchJson<HuggingFaceRowsResponse>(
			`https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(datasetId)}&config=${encodeURIComponent(configName)}&split=${encodeURIComponent(splitName)}&offset=0&length=${length}`
		);
		return (response.rows ?? [])
			.map((entry) => entry.row ?? {})
			.map(sanitizePreviewRow);
	} catch {
		return [];
	}
}

function sanitizePreviewRow(row: Record<string, unknown>): Record<string, unknown> {
	return Object.fromEntries(
		Object.entries(row).slice(0, 8).map(([key, value]) => {
			if (typeof value === 'string') {
				return [key, value];
			}
			if (typeof value === 'number' || typeof value === 'boolean' || value == null) {
				return [key, value];
			}
			return [key, JSON.stringify(value)];
		})
	);
}

async function loadHuggingFaceInfo(datasetId: string): Promise<{
	configName: string;
	configInfo: NonNullable<HuggingFaceInfoResponse['dataset_info']>[string];
	featureNames: string[];
}> {
	const info = await fetchJson<HuggingFaceInfoResponse>(
		`https://datasets-server.huggingface.co/info?dataset=${encodeURIComponent(datasetId)}`
	);
	const [configName, configInfo] = Object.entries(info.dataset_info ?? {})[0] ?? [];
	if (!configName || !configInfo) {
		throw new Error(`No dataset info was returned for ${datasetId}`);
	}

	const featureNames = Object.entries(configInfo.features ?? {})
		.filter(([, value]) => value?.dtype === 'string')
		.map(([key]) => key);

	return {
		configName,
		configInfo,
		featureNames
	};
}

function getSplitChecksum(
	checksumEntries: string[],
	splitName: string | null
): string | null {
	if (!splitName) return null;
	return checksumEntries.find((entry) => entry.endsWith(`/${splitName}.json`)) ?? null;
}

function normalizeSelectedTextFields(
	requestedFields: string[] | null | undefined,
	availableFields: string[]
): string[] {
	if (!requestedFields || requestedFields.length === 0) return availableFields;
	const available = new Set(availableFields);
	const selected = requestedFields
		.map((field) => field.trim())
		.filter((field, index, values) => field.length > 0 && values.indexOf(field) === index)
		.filter((field) => available.has(field));
	return selected.length > 0 ? selected : availableFields;
}

function suggestedCarvedValidationExamples(trainExamples: number | null): number | null {
	if (trainExamples == null || trainExamples < 2) return null;
	const tenPercent = Math.floor(trainExamples * 0.1);
	return Math.max(1, Math.min(tenPercent || 1, 5000));
}

function takeRows(rows: Record<string, unknown>[], maxExamples: number | null | undefined): Record<string, unknown>[] {
	if (maxExamples == null || maxExamples < 1 || maxExamples >= rows.length) {
		return rows;
	}
	return rows.slice(0, maxExamples);
}

function encodeExamples(
	rows: Record<string, unknown>[],
	renderExample: (row: Record<string, unknown>) => string
): { bytes: Uint8Array; examples: number } {
	const documents = rows
		.map(renderExample)
		.map((value) => value.trim())
		.filter(Boolean);
	const text = documents.join('\n\n');
	return {
		bytes: new TextEncoder().encode(text),
		examples: documents.length
	};
}

function summarizeVersion(row: DatasetVersionRow) {
	const trainer = getTrainerDefinition(row.trainer_key);
	const textFields = JSON.parse(row.text_fields_json) as string[];
	const recipe = resolveDatasetRecipe(row.source_ref, textFields, row.recipe_key);
	const samplePrompt = row.sample_prompt.trim() || recipe.samplePrompt;
	return {
		id: row.id,
		sourceType: row.source_type,
		sourceRef: row.source_ref,
		label: row.label,
		slug: row.slug,
		configName: row.config_name,
		revision: row.revision,
		recipeKey: row.recipe_key,
		trainerKey: row.trainer_key,
		trainerLabel: trainer.label,
		metricKey: trainer.metricKey,
		metricLabel: trainer.metricLabel,
		metricShortLabel: trainer.metricShortLabel,
		supportsInference: trainer.supportsInference,
		vocabSize: trainer.defaultVocabSize,
		modelFamily: row.model_family,
		textFields,
		recipeDescription: recipe.description,
		preprocessingSummary: recipe.preprocessingSummary,
		preprocessingSteps: recipe.preprocessingSteps,
		researchNotes: recipe.researchNotes,
		samplePrompt,
		trainExamples: row.train_examples,
		validationExamples: row.validation_examples,
		trainBytes: row.train_bytes,
		validationBytes: row.validation_bytes,
		maxTrainExamples: row.max_train_examples,
		maxValidationExamples: row.max_validation_examples,
		manifestPath: row.manifest_path,
		createdAt: row.created_at,
		isActive: row.is_active === 1
	};
}

function ensureLegacyDatasetVersion(): void {
	const trainPath = join(process.cwd(), 'static', 'data', 'train.bin');
	const validationPath = join(process.cwd(), 'static', 'data', 'val.bin');
	if (!existsSync(trainPath) || !existsSync(validationPath)) {
		return;
	}

	const existing = listDatasetVersionRows().find(
		(row) => row.source_type === 'legacy' && row.source_ref === LEGACY_SOURCE_REF
	);
	if (existing) {
		return;
	}

	const trainBytes = statSync(trainPath).size;
	const validationBytes = statSync(validationPath).size;
	const legacyDir = join(getDatasetRoot(), LEGACY_SLUG);
	mkdirSync(legacyDir, { recursive: true });
	const manifestPath = join(legacyDir, 'manifest.json');
	const hasActive = getActiveDatasetVersionRow() != null;
	const manifest = {
		label: LEGACY_LABEL,
		sourceType: 'legacy',
		sourceRef: LEGACY_SOURCE_REF,
		configName: 'legacy',
		revision: null,
		recipeKey: LEGACY_RECIPE_KEY,
		trainerKey: DEFAULT_TRAINER_KEY,
		modelFamily: 'byte-gpt',
		textFields: [],
		samplePrompt: resolveDatasetRecipe(LEGACY_SOURCE_REF, [], LEGACY_RECIPE_KEY).samplePrompt,
		trainExamples: 0,
		validationExamples: 0,
		trainBytes,
		validationBytes,
		maxTrainExamples: null,
		maxValidationExamples: null
	};
	writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

	insertDatasetVersion({
		source_type: 'legacy',
		source_ref: LEGACY_SOURCE_REF,
		label: LEGACY_LABEL,
		slug: LEGACY_SLUG,
		config_name: 'legacy',
		revision: null,
		recipe_key: LEGACY_RECIPE_KEY,
		trainer_key: DEFAULT_TRAINER_KEY,
		model_family: 'byte-gpt',
		text_fields_json: '[]',
		sample_prompt: manifest.samplePrompt,
		train_examples: 0,
		validation_examples: 0,
		train_bytes: trainBytes,
		validation_bytes: validationBytes,
		max_train_examples: null,
		max_validation_examples: null,
		manifest_path: manifestPath,
		train_path: trainPath,
		validation_path: validationPath,
		isActive: !hasActive
	});
}

export function listDatasetVersions() {
	ensureLegacyDatasetVersion();
	const versions = listDatasetVersionRows().map(summarizeVersion);
	const active = versions.find((version) => version.isActive) ?? null;
	return {
		versions,
		activeVersionId: active?.id ?? null
	};
}

export function getActiveDatasetVersionSummary() {
	ensureLegacyDatasetVersion();
	const row = getActiveDatasetVersionRow();
	return row ? summarizeVersion(row) : null;
}

export function setActiveDatasetVersion(versionId: number): ImportSummary {
	ensureLegacyDatasetVersion();
	setActiveDatasetVersionRow(versionId);
	return listDatasetVersions();
}

export function getDatasetSplitPath(versionId: number, split: 'train' | 'validation'): string {
	ensureLegacyDatasetVersion();
	const row = getDatasetVersionRowById(versionId);
	if (!row) {
		throw new Error(`Dataset version ${versionId} was not found.`);
	}
	return split === 'train' ? row.train_path : row.validation_path;
}

export function buildDefaultDatasetRecipeProposal(
	inspection: HuggingFaceDatasetInspection
): DatasetRecipeProposal {
	const recipeKey = inspection.recipeOptions[0]?.key ?? 'all-string-fields-v1';
	const recipe = resolveDatasetRecipe(inspection.datasetId, inspection.featureNames, recipeKey);
	const validationStrategy = inspection.hasValidationSplit ? 'huggingface' : 'carve-from-train';
	const maxValidationExamples = validationStrategy === 'huggingface'
		? inspection.validationExamples
		: suggestedCarvedValidationExamples(inspection.trainExamples);
	const maxTrainExamples = validationStrategy === 'huggingface'
		? inspection.trainExamples
		: (inspection.trainExamples != null && maxValidationExamples != null
			? Math.max(1, inspection.trainExamples - maxValidationExamples)
			: inspection.trainExamples);

	return {
		datasetId: inspection.datasetId,
		label: inspection.datasetId,
		trainerKey: recipe.trainerKey,
		modelFamily: recipe.modelFamily,
		recipeKey: recipe.key,
		textFields: recipe.textFields.length > 0 ? recipe.textFields : inspection.featureNames,
		maxTrainExamples,
		maxValidationExamples,
		validationStrategy,
		preprocessingSummary: recipe.preprocessingSummary,
		preprocessingSteps: recipe.preprocessingSteps,
		researchNotes: recipe.researchNotes,
		samplePrompt: recipe.samplePrompt,
		reasoning: 'Default proposal selected from dataset schema and built-in recipe heuristics.'
	};
}

export async function inspectHuggingFaceDataset(datasetId: string): Promise<HuggingFaceDatasetInspection> {
	const trimmedDatasetId = datasetId.trim();
	if (!trimmedDatasetId) {
		throw new Error('datasetId is required');
	}

	const { configName, configInfo, featureNames } = await loadHuggingFaceInfo(trimmedDatasetId);
	const trainSplit = configInfo.splits?.train ?? null;
	const validationSplit = configInfo.splits?.validation ?? null;
	const recipeOptions = listApplicableDatasetRecipes(trimmedDatasetId, featureNames);
	const [trainSamples, validationSamples] = await Promise.all([
		fetchSplitPreviewRows(trimmedDatasetId, configName, trainSplit?.name ?? 'train'),
		fetchSplitPreviewRows(trimmedDatasetId, configName, validationSplit?.name ?? null)
	]);

	const inspection = {
		datasetId: trimmedDatasetId,
		configName,
		featureNames,
		trainSplitName: trainSplit?.name ?? 'train',
		trainExamples: trainSplit?.num_examples ?? null,
		validationSplitName: validationSplit?.name ?? null,
		validationExamples: validationSplit?.num_examples ?? null,
		hasValidationSplit: Boolean(validationSplit),
		splitNames: Object.keys(configInfo.splits ?? {}),
		trainSamples,
		validationSamples,
		recipeOptions,
		defaultProposal: null as unknown as DatasetRecipeProposal
	};
	inspection.defaultProposal = buildDefaultDatasetRecipeProposal(inspection);
	return inspection;
}

export async function importDatasetFromHuggingFace(input: ImportRequest): Promise<ImportSummary> {
	const datasetId = input.datasetId.trim();
	if (!datasetId) {
		throw new Error('datasetId is required');
	}

	const { configName, configInfo, featureNames } = await loadHuggingFaceInfo(datasetId);
	const selectedTextFields = normalizeSelectedTextFields(input.textFields, featureNames);
	const recipe = resolveDatasetRecipe(datasetId, selectedTextFields, input.recipeKey);
	const checksumEntries = Object.keys(configInfo.download_checksums ?? {});
	const trainSplitName = configInfo.splits?.train?.name ?? 'train';
	const validationSplitName = configInfo.splits?.validation?.name ?? null;
	const trainChecksum = getSplitChecksum(checksumEntries, trainSplitName);
	const validationChecksum = getSplitChecksum(checksumEntries, validationSplitName);
	if (!trainChecksum) {
		throw new Error(`Expected a train split in ${datasetId}`);
	}

	const trainUrl = checksumUrlToDownloadUrl(trainChecksum);
	const trainRows = await fetchDatasetRows(trainUrl);

	let selectedTrainRows: Record<string, unknown>[];
	let selectedValidationRows: Record<string, unknown>[];

	if (validationChecksum) {
		const validationRows = await fetchDatasetRows(checksumUrlToDownloadUrl(validationChecksum));
		selectedTrainRows = takeRows(trainRows, input.maxTrainExamples ?? null);
		selectedValidationRows = takeRows(validationRows, input.maxValidationExamples ?? null);
	} else {
		const requestedValidationExamples = input.maxValidationExamples ?? null;
		if (requestedValidationExamples == null || requestedValidationExamples < 1) {
			throw new Error(
				`${datasetId} does not expose a validation split. Enter a validation count to carve from the train split.`
			);
		}

		const requestedTrainExamples = input.maxTrainExamples ?? (trainRows.length - requestedValidationExamples);
		const totalRequestedExamples = requestedTrainExamples + requestedValidationExamples;
		if (requestedTrainExamples < 1) {
			throw new Error('train count must be at least 1 after carving validation examples.');
		}
		if (totalRequestedExamples > trainRows.length) {
			throw new Error(
				`Requested ${requestedTrainExamples.toLocaleString()} train + ${requestedValidationExamples.toLocaleString()} validation examples, but train only has ${trainRows.length.toLocaleString()} rows.`
			);
		}

		const sourceRows = trainRows.slice(0, totalRequestedExamples);
		const validationStartIndex = sourceRows.length - requestedValidationExamples;
		selectedTrainRows = sourceRows.slice(0, validationStartIndex);
		selectedValidationRows = sourceRows.slice(validationStartIndex);
	}

	const encodedTrain = encodeExamples(selectedTrainRows, recipe.renderExample);
	const encodedValidation = encodeExamples(selectedValidationRows, recipe.renderExample);
	const samplePrompt = input.samplePrompt?.trim() || recipe.samplePrompt;

	const parsedTrainPath = parseHfPath(trainChecksum);
	const revision = parsedTrainPath?.revision ?? null;
	const slugBase = slugify(datasetId);
	const versionSlug = `${slugBase}-${Date.now()}`;
	const versionDir = join(getDatasetRoot(), versionSlug);
	mkdirSync(versionDir, { recursive: true });

	const trainPath = join(versionDir, 'train.bin');
	const validationPath = join(versionDir, 'val.bin');
	const manifestPath = join(versionDir, 'manifest.json');
	const manifest = {
		label: input.label?.trim() || datasetId,
		sourceType: 'huggingface',
		sourceRef: datasetId,
		configName,
		revision,
		recipeKey: recipe.key,
		trainerKey: recipe.trainerKey,
		modelFamily: recipe.modelFamily,
		textFields: recipe.textFields.length > 0 ? recipe.textFields : selectedTextFields,
		samplePrompt,
		trainExamples: encodedTrain.examples,
		validationExamples: encodedValidation.examples,
		trainBytes: encodedTrain.bytes.length,
		validationBytes: encodedValidation.bytes.length,
		maxTrainExamples: input.maxTrainExamples ?? null,
		maxValidationExamples: input.maxValidationExamples ?? null,
		remoteSplits: configInfo.splits ?? {}
	};

	writeFileSync(trainPath, encodedTrain.bytes);
	writeFileSync(validationPath, encodedValidation.bytes);
	writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

	const versionId = insertDatasetVersion({
		source_type: 'huggingface',
		source_ref: datasetId,
		label: input.label?.trim() || datasetId,
		slug: versionSlug,
		config_name: configName,
		revision,
		recipe_key: recipe.key,
		trainer_key: recipe.trainerKey,
		model_family: recipe.modelFamily,
		text_fields_json: JSON.stringify(manifest.textFields),
		sample_prompt: samplePrompt,
		train_examples: encodedTrain.examples,
		validation_examples: encodedValidation.examples,
		train_bytes: encodedTrain.bytes.length,
		validation_bytes: encodedValidation.bytes.length,
		max_train_examples: input.maxTrainExamples ?? null,
		max_validation_examples: input.maxValidationExamples ?? null,
		manifest_path: manifestPath,
		train_path: trainPath,
		validation_path: validationPath,
		isActive: true
	});

	setActiveDatasetVersionRow(versionId);
	return listDatasetVersions();
}
