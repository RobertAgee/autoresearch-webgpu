import { DataLoader } from '../data/loader';
import type { StepMetrics } from '../prepare';
import { insertExperiment, insertInference, insertLossCurve, updateWeightsPath } from '../db';
import { petname } from '../petname';
import { saveWeights } from '../weights';
import { DEFAULT_TRAINER_KEY } from '$lib/trainers';
import {
	BASELINE_RESEARCH_CONFIG,
	applyResearchProposal,
	buildTrainCodeFromConfig,
	extractResearchConfigFromCode,
	researchPhaseForIteration,
	normalizeResearchConfig,
	type ResearchConfig,
	type ResearchPhase,
	type ResearchProposal
} from './config';
import { evaluateResearchRun, summarizeEvalReport, type EvalReport } from './eval';
import { buildEvalSummary, type ExperimentEvalSummary } from './metrics';
import { parseClaudeResponse } from './parse';
import {
	buildSystemPrompt,
	buildUserPrompt,
	type ExperimentRecord,
	type ResearchDatasetContext
} from './prompt';
import type { ResearchEndpointProfile } from './providers';
import { executeTrainCode, type RunResult } from './sandbox';

type CandidateSpec = {
	config: ResearchConfig;
	reasoning: string;
	phase: ResearchPhase;
	changedKeys: (keyof ResearchConfig)[];
};

type Outcome = {
	config: ResearchConfig;
	code: string;
	reasoning: string;
	phase: ResearchPhase;
	stage: 'baseline' | 'quick-screen' | 'full-benchmark' | 'confirmation';
	changedKeys: (keyof ResearchConfig)[];
	result: RunResult;
	lossCurve: { step: number; loss: number }[];
	report: EvalReport | null;
};

export type ResearchCallbacks = {
	onExperimentStart?: (code: string, reasoning: string) => void;
	onStep?: (metrics: StepMetrics) => void;
	onExperimentDone?: (record: ExperimentRecord) => void;
	onError?: (error: string) => void;
	onCodeStream?: (text: string) => void;
	onReasoningStream?: (text: string) => void;
};

function compareNewestFirst(a: ExperimentRecord, b: ExperimentRecord): number {
	const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
	const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
	return (Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id) || b.id - a.id;
}

function extractScoreFromReasoning(reasoning: string): number {
	const match = reasoning.match(/\bscore=([-+]?\d+(?:\.\d+)?)\b/);
	return match ? Number(match[1]) : Number.NEGATIVE_INFINITY;
}

function outcomeScore(outcome: Outcome): number {
	if (outcome.result.error) return Number.NEGATIVE_INFINITY;
	if (!outcome.report) return -outcome.result.valBpb;
	if (!outcome.report.gated) return Number.NEGATIVE_INFINITY;
	return outcome.report.compositeScore;
}

function stageSeconds(stage: Outcome['stage'], fullSeconds: number): number {
	switch (stage) {
		case 'quick-screen':
			return Math.min(12, fullSeconds);
		case 'baseline':
		case 'full-benchmark':
		case 'confirmation':
		default:
			return fullSeconds;
	}
}

function keepReasoning(reasoning: string, phase: ResearchPhase, stage: Outcome['stage'], changedKeys: (keyof ResearchConfig)[], report: EvalReport | null): string {
	const delta = changedKeys.length > 0 ? changedKeys.join(',') : 'baseline';
	return `${reasoning} | phase=${phase} | stage=${stage} | delta=${delta} | ${summarizeEvalReport(report)}`;
}

export class ResearchController {
	history: ExperimentRecord[] = [];
	bestConfig: ResearchConfig = BASELINE_RESEARCH_CONFIG;
	bestCode: string = buildTrainCodeFromConfig(BASELINE_RESEARCH_CONFIG);
	bestBpb: number = Infinity;
	bestResearchScore = Number.NEGATIVE_INFINITY;
	running = false;
	lastError = '';
	trainSeconds = 30;
	profile: ResearchEndpointProfile | null = null;
	datasetContext: ResearchDatasetContext | null = null;
	private stopRequested = false;
	private runAbort: AbortController | null = null;
	private fetchAbort: AbortController | null = null;
	private iteration = 0;

	requestStopAfterCurrentRun() {
		this.stopRequested = true;
	}

	stopImmediately() {
		this.stopRequested = true;
		this.fetchAbort?.abort();
		this.runAbort?.abort();
	}

	stop() {
		this.stopImmediately();
	}

	stopCurrentRun() {
		this.runAbort?.abort();
	}

	async run(
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks = {}
	) {
		this.running = true;
		this.stopRequested = false;
		this.syncChampionFromHistory();

		if (this.history.length === 0) {
			await this.runBaseline(trainData, valData, callbacks);
		}

		while (!this.stopRequested) {
			const proposal = await this.getNextProposal(callbacks);
			if (this.stopRequested) break;
			if (!proposal) {
				callbacks.onError?.(this.lastError || 'Failed to get next config proposal from the research backend.');
				break;
			}
			await this.runCandidate(proposal, trainData, valData, callbacks);
			this.iteration += 1;
		}

		this.running = false;
	}

	private syncChampionFromHistory() {
		const autoHistory = this.history.filter((record) => record.source === 'auto');
		const accepted = [...autoHistory].sort(compareNewestFirst).find((record) => record.kept && !record.error);
		const champion = accepted ?? [...autoHistory]
			.sort(compareNewestFirst)
			.find((record) => !record.error && extractResearchConfigFromCode(record.code));

		if (champion) {
			const config = extractResearchConfigFromCode(champion.code);
			if (config) {
				this.bestConfig = config;
				this.bestCode = champion.code;
				this.bestBpb = champion.valBpb;
				this.bestResearchScore = champion.primaryScore ?? extractScoreFromReasoning(champion.reasoning);
			}
		} else {
			const configFromCode = extractResearchConfigFromCode(this.bestCode);
			this.bestConfig = configFromCode ?? BASELINE_RESEARCH_CONFIG;
			this.bestCode = buildTrainCodeFromConfig(this.bestConfig);
		}

		this.iteration = autoHistory.filter((record) => !record.rerunOf).length;
	}

	private async runBaseline(
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	) {
		const config = normalizeResearchConfig(this.bestConfig);
		const baseline = await this.trainAndEvaluate(
			config,
			'Baseline run with fixed config and benchmark evaluation.',
			'representation',
			'baseline',
			[],
			trainData,
			valData,
			callbacks
		);
		const kept = !baseline.result.error && (baseline.report?.gated ?? true);
		await this.persistOutcome(baseline, kept, callbacks);
		if (kept) {
			this.bestConfig = baseline.config;
			this.bestCode = baseline.code;
			this.bestBpb = baseline.result.valBpb;
			this.bestResearchScore = outcomeScore(baseline);
		}
	}

	private async runCandidate(
		candidate: CandidateSpec,
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	) {
		const quick = await this.trainAndEvaluate(
			candidate.config,
			candidate.reasoning,
			candidate.phase,
			'quick-screen',
			candidate.changedKeys,
			trainData,
			valData,
			callbacks
		);

		if (quick.result.error || (quick.report && !quick.report.gated)) {
			await this.persistOutcome(quick, false, callbacks);
			return;
		}

		const full = await this.trainAndEvaluate(
			candidate.config,
			candidate.reasoning,
			candidate.phase,
			'full-benchmark',
			candidate.changedKeys,
			trainData,
			valData,
			callbacks
		);

		if (full.result.error || (full.report && !full.report.gated)) {
			await this.persistOutcome(full, false, callbacks);
			return;
		}

		const challengerScore = outcomeScore(full);
		if (challengerScore <= this.bestResearchScore) {
			await this.persistOutcome(full, false, callbacks);
			return;
		}

		const benchmarkGroup = `confirm ${new Date().toISOString().replace('T', ' ').slice(0, 19)}`;
		const confirmation = await this.trainAndEvaluate(
			normalizeResearchConfig({ ...candidate.config, seed: candidate.config.seed + 1 }),
			`Confirmation rerun for candidate: ${candidate.reasoning}`,
			candidate.phase,
			'confirmation',
			[...candidate.changedKeys, 'seed'],
			trainData,
			valData,
			callbacks
		);

		const confirmationScore = outcomeScore(confirmation);
		const averageScore = averageFinite([challengerScore, confirmationScore]);
		const accepted = Number.isFinite(averageScore) && averageScore > this.bestResearchScore;

		const primaryRecord = await this.persistOutcome(full, accepted, callbacks, {
			benchmarkGroup
		});
		await this.persistOutcome(confirmation, false, callbacks, {
			rerunOf: primaryRecord.id,
			benchmarkGroup
		});

		if (accepted) {
			this.bestConfig = full.config;
			this.bestCode = full.code;
			this.bestBpb = full.result.valBpb;
			this.bestResearchScore = averageScore;
		}
	}

	private async trainAndEvaluate(
		config: ResearchConfig,
		reasoning: string,
		phase: ResearchPhase,
		stage: Outcome['stage'],
		changedKeys: (keyof ResearchConfig)[],
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	): Promise<Outcome> {
		const code = buildTrainCodeFromConfig(config);
		const stageReasoning = `${reasoning} [${stage}]`;
		callbacks.onExperimentStart?.(code, stageReasoning);

		this.runAbort = new AbortController();
		const lossCurve: { step: number; loss: number }[] = [];

		const result = await executeTrainCode(
			code,
			trainData,
			valData,
			stageSeconds(stage, this.trainSeconds),
			this.datasetContext?.trainerKey ?? DEFAULT_TRAINER_KEY,
			{
				signal: this.runAbort.signal,
				onStep: (metrics) => {
					lossCurve.push({ step: metrics.step, loss: metrics.loss });
					callbacks.onStep?.(metrics);
				}
			}
		);
		this.runAbort = null;

		const report = result.error
			? null
			: await evaluateResearchRun({
				params: result.params,
				forward: result.forward,
				vocabSize: result.vocabSize,
				seqLen: result.seqLen,
				context: this.datasetContext ?? undefined,
				valBpb: result.valBpb,
				mode: stage === 'quick-screen' ? 'quick' : 'full'
			});

		return {
			config,
			code,
			reasoning,
			phase,
			stage,
			changedKeys,
			result,
			lossCurve,
			report
		};
	}

	private async persistOutcome(
		outcome: Outcome,
		kept: boolean,
		callbacks: ResearchCallbacks,
		options: {
			rerunOf?: number | null;
			benchmarkGroup?: string | null;
		} = {}
	): Promise<ExperimentRecord> {
		const name = petname();
		const evalSummary: ExperimentEvalSummary | null = buildEvalSummary(
			outcome.report,
			outcome.stage,
			outcome.phase
		);
		const reasoning = keepReasoning(
			outcome.reasoning,
			outcome.phase,
			outcome.stage,
			outcome.changedKeys,
			outcome.report
		);

		const dbId = await insertExperiment({
			name,
			source: 'auto',
			code: outcome.code,
			datasetVersionId: this.datasetContext?.versionId ?? null,
			datasetLabel: this.datasetContext?.label ?? null,
			datasetSourceRef: this.datasetContext?.sourceRef ?? null,
			trainerKey: this.datasetContext?.trainerKey ?? DEFAULT_TRAINER_KEY,
			modelFamily: this.datasetContext?.modelFamily ?? 'byte-gpt',
			valBpb: outcome.result.valBpb,
			primaryScore: evalSummary?.primaryScore ?? null,
			elapsed: outcome.result.elapsed,
			totalSteps: outcome.result.totalSteps,
			reasoning,
			kept,
			lossCurve: outcome.lossCurve,
			error: outcome.result.error,
			evalSummary,
			rerunOf: options.rerunOf ?? null,
			benchmarkGroup: options.benchmarkGroup ?? null
		});

		await insertLossCurve(dbId, outcome.lossCurve);

		for (const sample of outcome.report?.samples ?? []) {
			await insertInference({
				experimentId: dbId,
				prompt: `[${outcome.stage}:${sample.label}] ${sample.prompt}`,
				output: sample.output,
				temperature: sample.temperature
			});
		}

		if (
			outcome.stage !== 'quick-screen' &&
			outcome.result.params &&
			Object.keys(outcome.result.params).length > 0
		) {
			(async () => {
				try {
					const weightsPath = await saveWeights(dbId, outcome.result.params);
					await updateWeightsPath(dbId, weightsPath);
				} catch (error) {
					console.error('Failed to save weights:', error);
				}
			})();
		}

		const record: ExperimentRecord = {
			id: dbId,
			name,
			source: 'auto',
			code: outcome.code,
			datasetVersionId: this.datasetContext?.versionId ?? null,
			datasetLabel: this.datasetContext?.label ?? null,
			datasetSourceRef: this.datasetContext?.sourceRef ?? null,
			trainerKey: this.datasetContext?.trainerKey ?? DEFAULT_TRAINER_KEY,
			modelFamily: this.datasetContext?.modelFamily ?? 'byte-gpt',
			valBpb: outcome.result.valBpb,
			primaryScore: evalSummary?.primaryScore ?? null,
			elapsed: outcome.result.elapsed,
			totalSteps: outcome.result.totalSteps,
			reasoning,
			kept,
			error: outcome.result.error,
			lossCurve: outcome.lossCurve,
			evalSummary,
			rerunOf: options.rerunOf ?? null,
			benchmarkGroup: options.benchmarkGroup ?? null,
			researchPhase: outcome.phase
		};

		this.history.push(record);
		callbacks.onExperimentDone?.(record);
		return record;
	}

	private async getNextProposal(callbacks: ResearchCallbacks): Promise<CandidateSpec | null> {
		const phase = researchPhaseForIteration(this.iteration);
		const systemPrompt = buildSystemPrompt(this.datasetContext ?? undefined, phase);
		const userPrompt = buildUserPrompt(
			this.history,
			this.bestConfig,
			this.bestResearchScore,
			phase,
			this.datasetContext ?? undefined
		);

		this.fetchAbort = new AbortController();
		try {
			const response = await fetch('/api/research', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					systemPrompt,
					userPrompt,
					stream: true,
					profile: this.profile
				}),
				signal: this.fetchAbort.signal
			});

			if (!response.ok) {
				const err = await response.text();
				this.lastError = `API ${response.status}: ${err.slice(0, 200)}`;
				return null;
			}

			const fullText = await this.consumeStream(response, callbacks);
			if (!fullText) {
				this.lastError = 'Empty response from the research backend';
				return null;
			}

			return this.parseProposal(fullText, phase, callbacks);
		} catch (error) {
			if (this.stopRequested) return null;
			this.lastError = `Fetch failed: ${error}`;
			return null;
		} finally {
			this.fetchAbort = null;
		}
	}

	private async consumeStream(response: Response, callbacks: ResearchCallbacks): Promise<string> {
		const reader = response.body!.getReader();
		const decoder = new TextDecoder();
		let fullText = '';
		let buffer = '';

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (!line.startsWith('data: ')) continue;
				const data = line.slice(6);
				if (data === '[DONE]') continue;

				try {
					const event = JSON.parse(data);
					if (event.type === 'text_delta' && event.text) {
						fullText += event.text;
						const reasoning = this.extractStreamingReasoning(fullText);
						if (reasoning) {
							callbacks.onReasoningStream?.(reasoning);
						}
					}
				} catch {
					// Ignore malformed SSE chunks.
				}
			}
		}

		return fullText;
	}

	private extractStreamingReasoning(partial: string): string | undefined {
		const match = partial.match(/"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"/);
		if (!match) return undefined;

		try {
			return JSON.parse(`"${match[1]}"`);
		} catch {
			return undefined;
		}
	}

	private parseProposal(
		text: string,
		phase: ResearchPhase,
		callbacks: ResearchCallbacks
	): CandidateSpec | null {
		const parsed = parseClaudeResponse(text);
		if (!parsed) {
			this.lastError = 'Could not parse research backend response';
			return null;
		}

		let proposal: ResearchProposal = parsed;
		if (!proposal.reasoning.trim()) {
			proposal = { ...proposal, reasoning: 'Small constrained challenger.' };
		}

		try {
			const applied = applyResearchProposal(this.bestConfig, proposal, phase);
			const code = buildTrainCodeFromConfig(applied.config);
			callbacks.onCodeStream?.(code);
			return {
				config: applied.config,
				reasoning: proposal.reasoning.trim(),
				phase,
				changedKeys: applied.changedKeys
			};
		} catch (error) {
			this.lastError = error instanceof Error ? error.message : String(error);
			return null;
		}
	}
}

function averageFinite(values: number[]): number {
	const finite = values.filter((value) => Number.isFinite(value));
	return finite.length > 0 ? finite.reduce((sum, value) => sum + value, 0) / finite.length : Number.NEGATIVE_INFINITY;
}
