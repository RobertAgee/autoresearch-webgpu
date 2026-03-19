import { BASELINE_CODE } from '$lib/research/baseline';

export type TrainerKey = 'byte-next-token-v1';

export type TrainerDefinition = {
	key: TrainerKey;
	label: string;
	description: string;
	metricKey: string;
	metricLabel: string;
	metricShortLabel: string;
	tokenizerLabel: string;
	objectiveSummary: string;
	defaultModelFamily: string;
	defaultVocabSize: number;
	supportsInference: boolean;
	researchNotes: string[];
	baselineCode: string;
};

const BYTE_NEXT_TOKEN_TRAINER: TrainerDefinition = {
	key: 'byte-next-token-v1',
	label: 'Byte Next-Token LM',
	description: 'Autoregressive next-token prediction over byte-level text corpora.',
	metricKey: 'val_bpb',
	metricLabel: 'validation bits-per-byte',
	metricShortLabel: 'bpb',
	tokenizerLabel: 'Byte-level tokenizer',
	objectiveSummary: 'Minimize next-token prediction loss on byte sequences.',
	defaultModelFamily: 'byte-gpt',
	defaultVocabSize: 256,
	supportsInference: true,
	researchNotes: [
		'This trainer predicts raw bytes, so tokenizer changes are out of scope unless preprocessing changes first.',
		'Shorter context and faster step throughput often beat larger models in the 30 second budget.',
		'Byte-level loss and sample quality can diverge, especially on structured symbolic data.'
	],
	baselineCode: BASELINE_CODE
};

const TRAINER_REGISTRY: Record<TrainerKey, TrainerDefinition> = {
	[BYTE_NEXT_TOKEN_TRAINER.key]: BYTE_NEXT_TOKEN_TRAINER
};

export const DEFAULT_TRAINER_KEY: TrainerKey = BYTE_NEXT_TOKEN_TRAINER.key;

export function listTrainers(): TrainerDefinition[] {
	return Object.values(TRAINER_REGISTRY);
}

export function getTrainerDefinition(key?: string | null): TrainerDefinition {
	if (!key) return BYTE_NEXT_TOKEN_TRAINER;
	return TRAINER_REGISTRY[key as TrainerKey] ?? BYTE_NEXT_TOKEN_TRAINER;
}

export function getBaselineCodeForTrainer(key?: string | null): string {
	return getTrainerDefinition(key).baselineCode;
}
