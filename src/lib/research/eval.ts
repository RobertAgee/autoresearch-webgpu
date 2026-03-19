import type { ForwardFn, Params } from '$lib/prepare';
import { sampleText } from '$lib/sample';
import type { ResearchDatasetContext } from './prompt';

export type BenchmarkSample = {
	label: string;
	prompt: string;
	temperature: number;
	output: string;
	structureScore: number;
	headerOk: boolean;
	hasK: boolean;
	hasL: boolean;
	terminated: boolean;
	promptFidelity: number;
	repetitionScore: number;
};

export type EvalReport = {
	label: string;
	sampleCount: number;
	gated: boolean;
	structureScore: number;
	headerRate: number;
	kRate: number;
	lRate: number;
	terminationRate: number;
	promptFidelity: number;
	repetitionScore: number;
	compositeScore: number;
	samples: BenchmarkSample[];
};

type EvalMode = 'quick' | 'full';

type RunEvalInput = {
	params: Params;
	forward: ForwardFn;
	vocabSize: number;
	seqLen: number;
	context?: ResearchDatasetContext;
	valBpb: number;
	mode: EvalMode;
};

function average(values: number[]): number {
	return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function containsOrderedSections(text: string): boolean {
	const controlStart = text.indexOf('[control]');
	const controlEnd = text.indexOf('[/control]');
	const abcStart = text.indexOf('[abc]');
	return controlStart >= 0 && controlEnd > controlStart && abcStart > controlEnd;
}

function extractAbcBody(text: string): string {
	const abcStart = text.indexOf('[abc]');
	if (abcStart < 0) return text;
	const afterStart = text.slice(abcStart + '[abc]'.length);
	const end = afterStart.indexOf('[/abc]');
	return (end >= 0 ? afterStart.slice(0, end) : afterStart).trim();
}

function repetitionScore(text: string): number {
	const body = extractAbcBody(text);
	if (!body) return 0;

	const lines = body
		.split('\n')
		.map((line) => line.trim())
		.filter(Boolean);

	if (lines.length === 0) return 0;

	const uniqueLineRatio = new Set(lines).size / lines.length;
	const normalized = body.replace(/\s+/g, ' ');
	const windows = new Map<string, number>();
	for (let i = 0; i + 12 <= normalized.length; i += 4) {
		const chunk = normalized.slice(i, i + 12);
		windows.set(chunk, (windows.get(chunk) ?? 0) + 1);
	}
	const maxWindowRepeats = Math.max(1, ...windows.values());
	const repeatPenalty = clamp((maxWindowRepeats - 1) / 6, 0, 1);
	return clamp(uniqueLineRatio * (1 - repeatPenalty), 0, 1);
}

function headerRate(text: string): boolean {
	const body = extractAbcBody(text);
	const lines = body.split('\n').map((line) => line.trim()).filter(Boolean);
	return lines.length > 0 && /^(X:|T:|M:|L:|Q:|R:|K:)/.test(lines[0]);
}

function promptFidelityScore(prompt: string, output: string): number {
	if (!output.startsWith(prompt)) return 0;
	const promptStructureOk = prompt.includes('[control]') ? containsOrderedSections(output) : 1;
	const abcIndex = output.indexOf('[abc]');
	if (abcIndex < 0) return 0;
	const afterAbc = output.slice(abcIndex + '[abc]'.length);
	const generatedBody = afterAbc.slice(Math.max(0, prompt.length - abcIndex - '[abc]'.length));
	const hasMusicalTokens = /[A-Ga-gzx|]/.test(generatedBody);
	return (promptStructureOk ? 0.5 : 0) + (hasMusicalTokens ? 0.5 : 0);
}

function structureMetrics(prompt: string, output: string): Omit<BenchmarkSample, 'label' | 'prompt' | 'temperature' | 'output'> {
	const hasHeader = headerRate(output);
	const hasK = /\bK:/.test(output);
	const hasL = /\bL:/.test(output);
	const terminated = output.includes('[/abc]');
	const promptFidelity = promptFidelityScore(prompt, output);
	const repetition = repetitionScore(output);
	const structureScore = average([
		hasHeader ? 1 : 0,
		hasK ? 1 : 0,
		hasL ? 1 : 0,
		terminated ? 1 : 0,
		promptFidelity,
		repetition
	]);

	return {
		structureScore,
		headerOk: hasHeader,
		hasK,
		hasL,
		terminated,
		promptFidelity,
		repetitionScore: repetition
	};
}

function abcPrompts(context?: ResearchDatasetContext, mode: EvalMode = 'full'): { label: string; prompt: string; temperature: number }[] {
	const prompts = [
		{
			label: 'reel-d',
			temperature: 0.3,
			prompt: '[control]\nT: brisk reel\nM:4/4\nR:reel\nL:1/8\nK:D\n[/control]\n\n[abc]\nX:1\nT:'
		},
		{
			label: 'jig-g',
			temperature: 0.45,
			prompt: '[control]\nT: lifty jig\nM:6/8\nR:jig\nL:1/8\nK:G\n[/control]\n\n[abc]\nX:1\n'
		},
		{
			label: 'hornpipe-dm',
			temperature: 0.55,
			prompt: '[control]\nT: crooked hornpipe\nM:4/4\nR:hornpipe\nL:1/8\nK:Dm\n[/control]\n\n[abc]\nX:1\n'
		},
		{
			label: 'waltz-g',
			temperature: 0.4,
			prompt: '[control]\nT: gentle waltz\nM:3/4\nR:waltz\nL:1/8\nK:G\n[/control]\n\n[abc]\nX:1\n'
		},
		{
			label: 'march-a',
			temperature: 0.6,
			prompt: '[control]\nT: clipped march\nM:2/4\nR:march\nL:1/8\nK:A\n[/control]\n\n[abc]\nX:1\n'
		}
	];

	if (context?.samplePrompt?.trim() && prompts.every((entry) => entry.prompt.startsWith('[control]'))) {
		return mode === 'quick' ? prompts.slice(0, 3) : prompts;
	}

	return mode === 'quick' ? prompts.slice(0, 3) : prompts;
}

function shouldRunAbcEval(context?: ResearchDatasetContext): boolean {
	return Boolean(context?.recipeKey?.includes('abc') || context?.sourceRef === 'sander-wood/irishman');
}

export async function evaluateResearchRun(input: RunEvalInput): Promise<EvalReport | null> {
	if (!shouldRunAbcEval(input.context)) {
		return null;
	}

	const prompts = abcPrompts(input.context, input.mode);
	const maxTokens = input.mode === 'quick' ? 180 : 260;
	const samples: BenchmarkSample[] = [];

	for (const entry of prompts) {
		const output = await sampleText(
			input.params,
			input.forward,
			input.vocabSize,
			input.seqLen,
			entry.prompt,
			maxTokens,
			entry.temperature
		);
		samples.push({
			label: entry.label,
			prompt: entry.prompt,
			temperature: entry.temperature,
			output,
			...structureMetrics(entry.prompt, output)
		});
	}

	const headerRateValue = average(samples.map((sample) => (sample.headerOk ? 1 : 0)));
	const kRate = average(samples.map((sample) => (sample.hasK ? 1 : 0)));
	const lRate = average(samples.map((sample) => (sample.hasL ? 1 : 0)));
	const terminationRate = average(samples.map((sample) => (sample.terminated ? 1 : 0)));
	const promptFidelity = average(samples.map((sample) => sample.promptFidelity));
	const repetition = average(samples.map((sample) => sample.repetitionScore));
	const structureScore = average(samples.map((sample) => sample.structureScore));
	const gated = headerRateValue >= 0.66 && kRate >= 1 && lRate >= 1 && terminationRate >= 0.34 && repetition >= 0.18;
	const compositeScore = structureScore * 100 - input.valBpb * 5;

	return {
		label: input.mode === 'quick' ? 'quick-screen' : 'full-benchmark',
		sampleCount: samples.length,
		gated,
		structureScore,
		headerRate: headerRateValue,
		kRate,
		lRate,
		terminationRate,
		promptFidelity,
		repetitionScore: repetition,
		compositeScore,
		samples
	};
}

export function summarizeEvalReport(report: EvalReport | null): string {
	if (!report) return 'sample_eval=not_applicable';

	return [
		`${report.label}`,
		`gate=${report.gated ? 'pass' : 'fail'}`,
		`score=${report.compositeScore.toFixed(2)}`,
		`struct=${report.structureScore.toFixed(2)}`,
		`header=${report.headerRate.toFixed(2)}`,
		`K=${report.kRate.toFixed(2)}`,
		`L=${report.lRate.toFixed(2)}`,
		`end=${report.terminationRate.toFixed(2)}`,
		`prompt=${report.promptFidelity.toFixed(2)}`,
		`repeat=${report.repetitionScore.toFixed(2)}`
	].join(' | ');
}
