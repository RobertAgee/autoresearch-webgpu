import type { DatasetRecipeProposal, HuggingFaceDatasetInspection } from './service';
import { buildDefaultDatasetRecipeProposal } from './service';
import { resolveDatasetRecipe } from './recipes';
import {
	DEFAULT_ANTHROPIC_URL,
	type ResearchEndpointProfile,
	type ResearchProvider
} from '$lib/research/providers';

type ResolvedProfile = {
	provider: ResearchProvider;
	baseUrl: string;
	apiKey: string;
	model: string;
};

function textContent(value: unknown): string {
	if (typeof value === 'string') return value;
	if (Array.isArray(value)) {
		return value
			.map((item) => {
				if (typeof item === 'string') return item;
				if (item && typeof item === 'object' && 'text' in item && typeof item.text === 'string') {
					return item.text;
				}
				return '';
			})
			.join('');
	}
	return '';
}

function resolveProfile(requestProfile: ResearchEndpointProfile | null | undefined): ResolvedProfile {
	if (
		requestProfile &&
		requestProfile.baseUrl.trim() &&
		requestProfile.apiKey.trim() &&
		requestProfile.model.trim()
	) {
		return {
			provider: requestProfile.provider === 'openai' ? 'openai' : 'anthropic',
			baseUrl: requestProfile.baseUrl.trim(),
			apiKey: requestProfile.apiKey.trim(),
			model: requestProfile.model.trim()
		};
	}

	throw new Error('No configured research backend was provided for recipe proposals.');
}

function buildUpstreamRequest(
	profile: ResolvedProfile,
	systemPrompt: string,
	userPrompt: string
): { url: string; init: RequestInit } {
	if (profile.provider === 'openai') {
		return {
			url: profile.baseUrl,
			init: {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${profile.apiKey}`
				},
				body: JSON.stringify({
					model: profile.model,
					stream: false,
					temperature: 0.2,
					max_tokens: 3000,
					messages: [
						{ role: 'system', content: systemPrompt },
						{ role: 'user', content: userPrompt }
					]
				})
			}
		};
	}

	return {
		url: profile.baseUrl || DEFAULT_ANTHROPIC_URL,
		init: {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'x-api-key': profile.apiKey,
				'anthropic-version': '2023-06-01'
			},
			body: JSON.stringify({
				model: profile.model,
				max_tokens: 3000,
				system: systemPrompt,
				messages: [{ role: 'user', content: userPrompt }]
			})
		}
	};
}

function extractResponseText(data: any, provider: ResearchProvider): string {
	if (provider === 'openai') {
		return textContent(data?.choices?.[0]?.message?.content);
	}

	return textContent(data?.content);
}

function extractJsonObject(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		const start = text.indexOf('{');
		const end = text.lastIndexOf('}');
		if (start >= 0 && end > start) {
			return JSON.parse(text.slice(start, end + 1));
		}
		throw new Error('The recipe proposal backend did not return valid JSON.');
	}
}

function parsePositiveInteger(value: unknown): number | null {
	if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value;
	if (typeof value === 'string') {
		const parsed = Number(value);
		return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
	}
	return null;
}

function validateProposal(
	inspection: HuggingFaceDatasetInspection,
	candidate: unknown
): DatasetRecipeProposal {
	const fallback = buildDefaultDatasetRecipeProposal(inspection);
	const proposed = candidate && typeof candidate === 'object' ? candidate as Record<string, unknown> : {};
	const allowedRecipeKeys = new Set(inspection.recipeOptions.map((option) => option.key));
	const requestedRecipeKey = typeof proposed.recipeKey === 'string' && allowedRecipeKeys.has(proposed.recipeKey)
		? proposed.recipeKey
		: fallback.recipeKey;

	const requestedTextFields = Array.isArray(proposed.textFields)
		? proposed.textFields.filter((field): field is string => typeof field === 'string')
		: fallback.textFields;
	const validTextFields = requestedTextFields.filter((field) => inspection.featureNames.includes(field));
	const textFields = validTextFields.length > 0 ? validTextFields : fallback.textFields;

	const recipe = resolveDatasetRecipe(inspection.datasetId, textFields, requestedRecipeKey);
	const validationStrategy = inspection.hasValidationSplit
		? 'huggingface'
		: 'carve-from-train';

	let maxValidationExamples = parsePositiveInteger(proposed.maxValidationExamples) ?? fallback.maxValidationExamples;
	let maxTrainExamples = parsePositiveInteger(proposed.maxTrainExamples) ?? fallback.maxTrainExamples;

	if (validationStrategy === 'huggingface') {
		if (inspection.trainExamples != null && maxTrainExamples != null) {
			maxTrainExamples = Math.min(maxTrainExamples, inspection.trainExamples);
		}
		if (inspection.validationExamples != null && maxValidationExamples != null) {
			maxValidationExamples = Math.min(maxValidationExamples, inspection.validationExamples);
		}
	} else {
		if (maxValidationExamples == null) {
			maxValidationExamples = fallback.maxValidationExamples;
		}
		if (inspection.trainExamples != null && maxValidationExamples != null) {
			const maxAvailableTrain = inspection.trainExamples - maxValidationExamples;
			maxTrainExamples = maxTrainExamples == null
				? maxAvailableTrain
				: Math.min(maxTrainExamples, maxAvailableTrain);
		}
	}

	return {
		datasetId: inspection.datasetId,
		label: typeof proposed.label === 'string' && proposed.label.trim() ? proposed.label.trim() : fallback.label,
		trainerKey: recipe.trainerKey,
		modelFamily: recipe.modelFamily,
		recipeKey: recipe.key,
		textFields: recipe.textFields.length > 0 ? recipe.textFields : textFields,
		maxTrainExamples,
		maxValidationExamples,
		validationStrategy,
		preprocessingSummary: recipe.preprocessingSummary,
		preprocessingSteps: recipe.preprocessingSteps,
		researchNotes: recipe.researchNotes,
		samplePrompt: typeof proposed.samplePrompt === 'string' && proposed.samplePrompt.trim()
			? proposed.samplePrompt.trim()
			: recipe.samplePrompt,
		reasoning: typeof proposed.reasoning === 'string' && proposed.reasoning.trim()
			? proposed.reasoning.trim()
			: fallback.reasoning
	};
}

function buildSystemPrompt(inspection: HuggingFaceDatasetInspection): string {
	return `You design deterministic dataset preprocessing plans for local ML experiments.

Return only one JSON object. Do not include markdown.

You are choosing among existing recipe keys, not inventing arbitrary code.
Only use recipeKey values from the provided options.
Only use textFields from the provided string feature names.
Preserve reproducibility: prefer stable, explicit structure over clever transforms.

Required JSON fields:
- label: short human-readable dataset version label
- recipeKey: chosen recipe key
- textFields: ordered array of selected string fields
- samplePrompt: short prompt prefix matching the chosen recipe format
- maxTrainExamples: positive integer or null
- maxValidationExamples: positive integer or null
- reasoning: concise explanation
`;
}

function buildUserPrompt(
	inspection: HuggingFaceDatasetInspection,
	fallback: DatasetRecipeProposal
): string {
	return JSON.stringify({
		datasetId: inspection.datasetId,
		configName: inspection.configName,
		featureNames: inspection.featureNames,
		trainSplitName: inspection.trainSplitName,
		trainExamples: inspection.trainExamples,
		validationSplitName: inspection.validationSplitName,
		validationExamples: inspection.validationExamples,
		hasValidationSplit: inspection.hasValidationSplit,
		splitNames: inspection.splitNames,
		recipeOptions: inspection.recipeOptions,
		trainSamples: inspection.trainSamples,
		validationSamples: inspection.validationSamples,
		defaultProposal: fallback
	}, null, 2);
}

export async function proposeDatasetRecipeWithBackend(
	inspection: HuggingFaceDatasetInspection,
	requestProfile: ResearchEndpointProfile | null | undefined
): Promise<DatasetRecipeProposal> {
	const fallback = buildDefaultDatasetRecipeProposal(inspection);
	const profile = resolveProfile(requestProfile);
	const upstream = buildUpstreamRequest(profile, buildSystemPrompt(inspection), buildUserPrompt(inspection, fallback));
	const response = await fetch(upstream.url, upstream.init);
	if (!response.ok) {
		throw new Error(await response.text());
	}

	const data = await response.json();
	const text = extractResponseText(data, profile.provider);
	const parsed = extractJsonObject(text);
	return validateProposal(inspection, parsed);
}
