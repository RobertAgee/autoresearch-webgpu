export type DatasetVersionSummary = {
	id: number;
	sourceType: 'huggingface' | 'legacy';
	sourceRef: string;
	label: string;
	slug: string;
	configName: string;
	revision: string | null;
	recipeKey: string;
	trainerKey: string;
	trainerLabel: string;
	metricKey: string;
	metricLabel: string;
	metricShortLabel: string;
	supportsInference: boolean;
	vocabSize: number;
	modelFamily: string;
	textFields: string[];
	recipeDescription: string;
	preprocessingSummary: string;
	preprocessingSteps: string[];
	researchNotes: string[];
	samplePrompt: string;
	trainExamples: number;
	validationExamples: number;
	trainBytes: number;
	validationBytes: number;
	maxTrainExamples: number | null;
	maxValidationExamples: number | null;
	manifestPath: string;
	createdAt: string;
	isActive: boolean;
};

export type DatasetCatalogResponse = {
	versions: DatasetVersionSummary[];
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
	recipeOptions: {
		key: string;
		label: string;
		description: string;
		trainerKey: string;
		modelFamily: string;
		textFields: string[];
		preprocessingSummary: string;
		preprocessingSteps: string[];
		researchNotes: string[];
		samplePrompt: string;
	}[];
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

async function readJson<T>(response: Response): Promise<T> {
	if (!response.ok) {
		throw new Error(await response.text());
	}
	return response.json() as Promise<T>;
}

export async function listDatasetVersions(): Promise<DatasetCatalogResponse> {
	return readJson<DatasetCatalogResponse>(await fetch('/api/datasets'));
}

export async function inspectHuggingFaceDataset(datasetId: string): Promise<HuggingFaceDatasetInspection> {
	const url = new URL('/api/datasets', window.location.origin);
	url.searchParams.set('inspect', 'huggingface');
	url.searchParams.set('datasetId', datasetId);
	return readJson<HuggingFaceDatasetInspection>(await fetch(url));
}

export async function importDatasetFromHuggingFace(input: {
	datasetId: string;
	label?: string;
	recipeKey?: string;
	textFields?: string[];
	samplePrompt?: string;
	maxTrainExamples?: number | null;
	maxValidationExamples?: number | null;
}): Promise<DatasetCatalogResponse> {
	return readJson<DatasetCatalogResponse>(
		await fetch('/api/datasets', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				action: 'importFromHuggingFace',
				...input
			})
		})
	);
}

export async function proposeDatasetRecipe(input: {
	inspection: HuggingFaceDatasetInspection;
	profile?: import('./research/providers').ResearchEndpointProfile | null;
}): Promise<{ proposal: DatasetRecipeProposal }> {
	return readJson<{ proposal: DatasetRecipeProposal }>(
		await fetch('/api/datasets', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				action: 'proposeRecipe',
				...input
			})
		})
	);
}

export async function setActiveDatasetVersion(versionId: number): Promise<DatasetCatalogResponse> {
	return readJson<DatasetCatalogResponse>(
		await fetch('/api/datasets', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				action: 'setActive',
				versionId
			})
		})
	);
}
