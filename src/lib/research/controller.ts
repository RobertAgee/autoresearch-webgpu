import { DataLoader } from '../data/loader';
import { executeTrainCode, type RunResult } from './sandbox';
import type { StepMetrics } from '../prepare';
import { petname } from '../petname';
import { insertExperiment, insertLossCurve, updateWeightsPath } from '../db';
import { saveWeights } from '../weights';
import { buildSystemPrompt, buildUserPrompt, type ExperimentRecord } from './prompt';
import { BASELINE_CODE } from './baseline';

export type ResearchCallbacks = {
	onExperimentStart?: (code: string, reasoning: string) => void;
	onStep?: (metrics: StepMetrics) => void;
	onExperimentDone?: (record: ExperimentRecord) => void;
	onError?: (error: string) => void;
};

export class ResearchController {
	history: ExperimentRecord[] = [];
	bestCode: string = BASELINE_CODE;
	bestBpb: number = Infinity;
	running: boolean = false;
	lastError = '';
	trainSeconds = 30;
	private stopRequested = false;
	private runAbort: AbortController | null = null;
	private fetchAbort: AbortController | null = null;

	stop() {
		this.stopRequested = true;
		this.fetchAbort?.abort();
		this.runAbort?.abort();
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

		// Run baseline if no history
		if (this.history.length === 0) {
			await this.runExperiment(
				this.bestCode,
				'Baseline run with default architecture.',
				trainData, valData, callbacks
			);
		}

		while (!this.stopRequested) {
			const proposal = await this.getNextCode();
			if (!proposal) {
				callbacks.onError?.(this.lastError || 'Failed to get next code from Claude.');
				break;
			}
			await this.runExperiment(
				proposal.code,
				proposal.reasoning,
				trainData, valData, callbacks
			);
		}

		this.running = false;
	}

	private async runExperiment(
		code: string,
		reasoning: string,
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	) {
		callbacks.onExperimentStart?.(code, reasoning);

		this.runAbort = new AbortController();
		const lossCurve: { step: number; loss: number }[] = [];

		const result = await executeTrainCode(code, trainData, valData, this.trainSeconds, {
			signal: this.runAbort.signal,
			onStep(m) {
				lossCurve.push({ step: m.step, loss: m.loss });
				callbacks.onStep?.(m);
			}
		});
		this.runAbort = null;

		const kept = result.valBpb < this.bestBpb && !result.error;
		if (kept) {
			this.bestBpb = result.valBpb;
			this.bestCode = code;
		}

		const expName = petname();
		const dbId = await insertExperiment({
			name: expName,
			source: 'auto',
			code,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			lossCurve,
			error: result.error,
		});

		await insertLossCurve(dbId, lossCurve);

		// Save weights in background
		if (result.params && Object.keys(result.params).length > 0) {
			(async () => {
				try {
					const weightsPath = await saveWeights(dbId, result.params);
					await updateWeightsPath(dbId, weightsPath);
				} catch (_) {}
			})();
		}

		const record: ExperimentRecord = {
			id: dbId,
			name: expName,
			source: 'auto',
			code,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			error: result.error,
			lossCurve
		};

		this.history.push(record);
		callbacks.onExperimentDone?.(record);
	}

	private async getNextCode(): Promise<{ code: string; reasoning: string } | null> {
		const systemPrompt = buildSystemPrompt();
		const userPrompt = buildUserPrompt(this.history, this.bestCode, this.bestBpb);

		this.fetchAbort = new AbortController();
		try {
			const response = await fetch('/api/research', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ systemPrompt, userPrompt }),
				signal: this.fetchAbort.signal
			});

			if (!response.ok) {
				const err = await response.text();
				this.lastError = `API ${response.status}: ${err.slice(0, 200)}`;
				return null;
			}

			const data = await response.json();
			if (data.error) {
				this.lastError = `API error: ${data.error}`;
				return null;
			}

			if (!data.code || typeof data.code !== 'string') {
				this.lastError = 'No code in API response';
				return null;
			}

			return {
				code: data.code,
				reasoning: data.reasoning || 'No reasoning provided.'
			};
		} catch (e) {
			if (this.stopRequested) return null;
			this.lastError = `Fetch failed: ${e}`;
			return null;
		} finally {
			this.fetchAbort = null;
		}
	}
}
