import { getTrainerDefinition } from '$lib/trainers';
import {
	extractResearchConfigFromCode,
	phaseFields,
	summarizeResearchConfig,
	type ResearchConfig,
	type ResearchPhase
} from './config';
import type { ExperimentEvalSummary } from './metrics';

export type ExperimentRecord = {
	id: number;
	projectId?: number | null;
	name: string;
	source: 'manual' | 'auto';
	code: string;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	error?: string;
	lossCurve?: { step: number; loss: number }[];
	rerunOf?: number | null;
	benchmarkGroup?: string | null;
	createdAt?: string;
	datasetVersionId?: number | null;
	datasetLabel?: string | null;
	datasetSourceRef?: string | null;
	trainerKey?: string | null;
	modelFamily?: string | null;
	researchPhase?: string | null;
	primaryScore?: number | null;
	evalSummary?: ExperimentEvalSummary | null;
};

export type ResearchDatasetContext = {
	versionId?: number | null;
	label: string;
	sourceRef: string;
	recipeKey: string;
	recipeDescription: string;
	preprocessingSummary: string;
	preprocessingSteps: string[];
	researchNotes: string[];
	samplePrompt: string;
	trainerKey: string;
	modelFamily: string;
	vocabSize: number;
	trainBytes: number;
	validationBytes: number;
	textFields: string[];
};

function experimentTimeValue(exp: ExperimentRecord): number {
	if (!exp.createdAt) return exp.id;
	const parsed = Date.parse(exp.createdAt);
	return Number.isFinite(parsed) ? parsed : exp.id;
}

function compareNewestFirst(a: ExperimentRecord, b: ExperimentRecord): number {
	return experimentTimeValue(b) - experimentTimeValue(a) || b.id - a.id;
}

function compareBestFirst(a: ExperimentRecord, b: ExperimentRecord): number {
	return Number(b.kept) - Number(a.kept) || a.valBpb - b.valBpb || compareNewestFirst(a, b);
}

function formatValBpb(value: number): string {
	return Number.isFinite(value) ? value.toFixed(4) : 'Infinity';
}

function summarizeExperiment(exp: ExperimentRecord): string {
	const badge = exp.kept ? 'KEPT' : 'DISCARDED';
	let msg = `#${exp.id} [${badge}] val_bpb=${formatValBpb(exp.valBpb)} (${exp.totalSteps} steps, ${(exp.elapsed / 1000).toFixed(1)}s)`;
	if (exp.rerunOf) {
		msg += ` [rerun of #${exp.rerunOf}]`;
	}
	const config = extractResearchConfigFromCode(exp.code);
	if (config) {
		msg += `\n  config: ${summarizeResearchConfig(config)}`;
	}
	if (exp.reasoning) {
		msg += `\n  ${exp.reasoning}`;
	}
	if (exp.error) {
		msg += `\n  ERROR: ${exp.error}`;
	}
	return msg;
}

/**
 * System prompt for constrained research proposals.
 */
export function buildSystemPrompt(context: ResearchDatasetContext | undefined, phase: ResearchPhase): string {
	const trainer = getTrainerDefinition(context?.trainerKey);
	const datasetLabel = context?.label ?? 'the active local dataset';
	const datasetSourceRef = context?.sourceRef ?? 'local import';
	const trainBytes = context?.trainBytes ?? 0;
	const validationBytes = context?.validationBytes ?? 0;
	const textFieldSummary = context?.textFields?.length
		? context.textFields.join(', ')
		: 'dataset text fields';
	const vocabSize = context?.vocabSize ?? trainer.defaultVocabSize;
	const recipeDescription = context?.recipeDescription ?? 'Generic text packing for the active dataset.';
	const preprocessingSummary = context?.preprocessingSummary ?? 'No recipe-specific preprocessing summary was provided.';
	const preprocessingSteps = context?.preprocessingSteps?.length
		? context.preprocessingSteps.map((step) => `- ${step}`).join('\n')
		: '- No explicit preprocessing steps were provided.';
	const trainerNotes = trainer.researchNotes.length
		? trainer.researchNotes.map((note) => `- ${note}`).join('\n')
		: '- No trainer-specific notes were provided.';
	const datasetNotes = context?.researchNotes?.length
		? context.researchNotes.map((note) => `- ${note}`).join('\n')
		: '- No dataset-specific notes were provided.';
	const samplePrompt = context?.samplePrompt?.trim()
		? context.samplePrompt
		: '(none)';
	const allowedFields = phaseFields(phase).join(', ');

	return `You are an autonomous ML researcher. You are NOT allowed to rewrite the training program.

Your goal: improve usable sample quality for this dataset while still caring about ${trainer.metricLabel} (${trainer.metricKey}).
The runtime now assembles the training code from a constrained config schema and rejects unusable symbolic music even if ${trainer.metricKey} improves.

## Environment
- You are running on a web interface in the browser with WebGPU (Apple M-series GPU, ~8GB shared memory)
- Trainer: ${trainer.label} (${trainer.key})
- Objective: ${trainer.objectiveSummary}
- ${trainer.tokenizerLabel} (vocab_size=${vocabSize})
- Active dataset: ${datasetLabel} (${datasetSourceRef})
- Dataset recipe: ${context?.recipeKey ?? 'unknown'} - ${recipeDescription}
- Imported text fields: ${textFieldSummary}
- Current local corpus size: train=${trainBytes} bytes, validation=${validationBytes} bytes
- Multi-fidelity pipeline:
  - quick screen: ~12 seconds training + lightweight sample checks
  - full eval: ~30 seconds training + benchmark prompt evaluation
  - confirmation rerun: only for provisional winners
- Practical limit: ~300K parameters. Bigger models = fewer steps in the budget, which often hurts.
- The sweet spot is small, fast models that get many training steps in 30 seconds.
- Current search phase: ${phase}
- Allowed fields this phase: ${allowedFields}
- Maximum mutation size this phase: ${phase === 'mixed' ? '4 fields' : '3 fields'}

## Trainer-specific guidance
${trainerNotes}

## Dataset-specific guidance
${datasetNotes}

## Preprocessing plan
- Summary: ${preprocessingSummary}
${preprocessingSteps}
- Preferred sample prefix: ${samplePrompt}

## Critical objective
- For ABC datasets, valid structure, clean termination, prompt-structure fidelity, and low collapse matter more than tiny ${trainer.metricShortLabel} gains.
- Proposals that fail basic ABC structure gates will be discarded.
- Most proposals should be small explicit edits to the current champion, not broad jumps.

## Output format
Respond with ONLY a JSON object:
{
  "reasoning": "one sentence explaining the hypothesis",
  "changes": {
    "fieldName": value
  }
}

Rules:
- Change only allowed fields for the current phase: ${allowedFields}
- Change at most ${phase === 'mixed' ? '4' : '3'} fields
- Prefer 1-2 field edits unless there is a strong reason not to
- Do not include code
- Do not include markdown fences
- Do not restate unchanged config values
- If the current champion already looks good, still propose a small challenger rather than returning an empty diff`;
}

export function buildUserPrompt(
	history: ExperimentRecord[],
	bestConfig: ResearchConfig,
	bestScore: number,
	phase: ResearchPhase,
	context?: ResearchDatasetContext
): string {
	const trainer = getTrainerDefinition(context?.trainerKey);
	let msg = `Current champion config (phase=${phase}):\n`;
	msg += `${summarizeResearchConfig(bestConfig)}\n`;
	msg += `Champion research score: ${Number.isFinite(bestScore) ? bestScore.toFixed(2) : '-Infinity'}\n`;
	msg += `Champion ${trainer.metricKey}: ${formatValBpb(history.find((exp) => exp.kept)?.valBpb ?? Infinity)}\n\n`;

	if (history.length > 0) {
		const valid = history.filter((exp) => !exp.error && Number.isFinite(exp.valBpb));
		const bestByCode = new Map<string, ExperimentRecord>();
		for (const exp of [...valid].sort(compareBestFirst)) {
			if (!bestByCode.has(exp.code)) {
				bestByCode.set(exp.code, exp);
			}
		}

		const topPerformers = [...bestByCode.values()].sort(compareBestFirst).slice(0, 5);
		if (topPerformers.length > 0) {
			msg += 'Top performers (best unique code first):\n';
			for (const exp of topPerformers) {
				msg += `\n${summarizeExperiment(exp)}`;
			}
			msg += '\n\n';
		}

		msg += 'Recent experiment activity (newest first by created_at):\n';
		const recent = [...history].sort(compareNewestFirst).slice(0, 10);
		for (const exp of recent) {
			msg += `\n${summarizeExperiment(exp)}`;
		}
		msg += '\n';
	}

	if (context) {
		msg += '\nDataset operating notes:\n';
		msg += `- recipe: ${context.recipeKey}\n`;
		msg += `- preprocessing: ${context.preprocessingSummary}\n`;
		if (context.researchNotes.length > 0) {
			for (const note of context.researchNotes) {
				msg += `- note: ${note}\n`;
			}
		}
		if (context.samplePrompt.trim()) {
			msg += `- sample prefix: ${JSON.stringify(context.samplePrompt)}\n`;
		}
	}

	msg += `\nPropose the next challenger as a constrained config delta. Respond with ONLY the JSON object.`;
	return msg;
}
