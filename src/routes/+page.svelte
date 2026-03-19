<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DataLoader } from '$lib/data/loader';
	import { executeTrainCode, type RunResult } from '$lib/research/sandbox';
	import type { StepMetrics, ForwardFn, Params } from '$lib/prepare';
	import { sampleText } from '$lib/sample';
	import { ResearchController } from '$lib/research/controller';
	import type { ExperimentRecord } from '$lib/research/prompt';
	import {
		DEFAULT_TRAINER_KEY,
		getBaselineCodeForTrainer,
		getTrainerDefinition
	} from '$lib/trainers';
	import {
		getDb, insertExperiment, insertInference, insertLossCurve,
		getInferencesForExperiment, clearAllData, deleteExperiments,
		exportCsvZip, importCsvZip, updateWeightsPath, getAllExperimentRecords, type InferenceRow
	} from '$lib/db';
	import {
		inspectHuggingFaceDataset,
		importDatasetFromHuggingFace,
		listDatasetVersions,
		proposeDatasetRecipe,
		setActiveDatasetVersion,
		type DatasetRecipeProposal,
		type DatasetVersionSummary,
		type HuggingFaceDatasetInspection
	} from '$lib/datasets';
	import { saveWeights, loadWeights } from '$lib/weights';
	import LossChart from '$lib/components/LossChart.svelte';
	import CodeEditor from '$lib/components/CodeEditor.svelte';
	import Leaderboard from '$lib/components/Leaderboard.svelte';
	import ResultsSummary from '$lib/components/ResultsSummary.svelte';
	import EndpointManager from '$lib/components/EndpointManager.svelte';
	import { petname } from '$lib/petname';
	import { isConfiguredProfile, type ResearchEndpointProfile } from '$lib/research/providers';
	import {
		formatExperimentMetricValue,
		getDatasetMetricColumns,
		getExperimentPrimaryMetric
	} from '$lib/research/metrics';
	import {
		loadActiveResearchProfileId,
		loadResearchProfiles,
		saveActiveResearchProfileId,
		saveResearchProfiles
	} from '$lib/research/profile-store';
	import {
		createExperimentWorkspace,
		listExperimentWorkspaces,
		type ExperimentWorkspaceRecord
	} from '$lib/workspaces';

	type ResultsScope = 'all' | 'adHoc' | 'reruns' | 'benchmark' | 'family';

	let gpuStatus = $state<WebGPUStatus | null>(null);
	let code = $state(getBaselineCodeForTrainer(DEFAULT_TRAINER_KEY));
	let running = $state(false);
	let lossData = $state<{ step: number; loss: number }[]>([]);
	let status = $state<string>('initializing');
	let stopIntent = $state<'none' | 'graceful' | 'immediate'>('none');
	let sampling = $state(false);
	let mode = $state<'manual' | 'research'>('research');
	let experiments = $state<ExperimentRecord[]>([]);
	let currentReasoning = $state('');
	let experimentName = $state(petname());
	let listMode = $state<'leaderboard' | 'current'>('leaderboard');
	let datasetVersions = $state<DatasetVersionSummary[]>([]);
	let activeDatasetVersionId = $state<number | null>(null);
	let datasetImportId = $state('sander-wood/irishman');
	let datasetImportMaxTrain = $state('');
	let datasetImportMaxValidation = $state('');
	let datasetImportInfo = $state<HuggingFaceDatasetInspection | null>(null);
	let datasetImportInfoBusy = $state(false);
	let datasetImportInfoError = $state('');
	let datasetRecipeDraft = $state<DatasetRecipeProposal | null>(null);
	let datasetProposalBusy = $state(false);
	let datasetProposalError = $state('');
	let datasetInspectorOpen = $state(true);
	let activeDatasetReviewOpen = $state(true);
	let datasetPrepConfirmedVersionId = $state<number | null>(null);
	let datasetImportSuggestedTrain = $state('');
	let datasetImportSuggestedValidation = $state('');
	let datasetImportDefaultsSourceId = $state('');
	let datasetBusy = $state(false);

	// Inference state — we need to store forward fn + params for sampling
	let prompt = $state('');
	let temperature = $state(0.8);
	let selectedExpId = $state<number | null>(null);
	let inferences = $state<InferenceRow[]>([]);
	let inferenceIdx = $state(0);
	let streamingOutput = $state('');
	let currentRunName = $state('');
	let trainAbort: AbortController | null = null;
	let inProgressExp = $state<ExperimentRecord | null>(null);
	let waitingForRecommendation = $state(false);
	let researchProfiles = $state<ResearchEndpointProfile[]>([]);
	let selectedResearchProfileId = $state<string | null>(null);
	let importing = $state(false);
	let importInput = $state<HTMLInputElement | null>(null);
	let allExperimentOptions = $state<ExperimentRecord[]>([]);
	let selectionMode = $state(false);
	let selectedBatchIds = $state<number[]>([]);
	let workspaceTab = $state<'chart' | 'code' | 'inference'>('chart');
	let setupTab = $state<'experiments' | 'createExperiment' | 'importDataset' | 'importResults' | 'exportResults' | 'research'>('experiments');
	let leaderboardSort = $state<'bpb' | 'newest' | 'oldest' | 'steps' | 'name'>('bpb');
	let resultsScope = $state<ResultsScope>('all');
	let selectedBenchmarkGroup = $state<string | null>(null);
	let selectedModelFamily = $state<string | null>(null);
	let chartSeriesMode = $state<'all' | 'focus'>('all');
	let chartScaleMode = $state<'fit' | 'trim' | 'manual'>('fit');
	let chartYMinInput = $state('');
	let chartYMaxInput = $state('');
	let workspaces = $state<ExperimentWorkspaceRecord[]>([]);
	let currentWorkspaceId = $state<number | null>(null);
	let createExperimentSourceType = $state<'dataset' | 'results'>('dataset');
	let createExperimentDatasetId = $state('');
	let createExperimentBaseExperimentId = $state('');
	let createExperimentName = $state('');
	let createExperimentReadme = $state('');
	let createExperimentNotes = $state('');
	let createExperimentBusy = $state(false);
	let exportWorkspaceId = $state('');

	// In-memory loaded model state for inference
	type LoadedModel = { forward: ForwardFn; params: Params; vocabSize: number; seqLen: number; expId: number };
	let loadedModel = $state<LoadedModel | null>(null);

	let allExperiments = $derived(
		inProgressExp ? [...experiments, inProgressExp] : experiments
	);
	let selectedBatchExperiments = $derived(
		experiments.filter((exp) => selectedBatchIds.includes(exp.id))
	);
	let visibleLeaderboardExperiments = $derived.by(() => {
		return allExperiments.filter((exp) => {
			switch (resultsScope) {
				case 'adHoc':
					return !exp.benchmarkGroup;
				case 'reruns':
					return Boolean(exp.rerunOf);
				case 'benchmark':
					return selectedBenchmarkGroup == null
						? Boolean(exp.benchmarkGroup)
						: exp.benchmarkGroup === selectedBenchmarkGroup;
				case 'family':
					return (exp.modelFamily?.trim() || 'byte-gpt') === (selectedModelFamily ?? 'byte-gpt');
				case 'all':
				default:
					return true;
			}
		});
	});
	let activeMetricColumns = $derived(
		getDatasetMetricColumns(activeDataset ? { recipeKey: activeDataset.recipeKey } : null)
	);
	let selectedPrimaryMetric = $derived(
		selectedExp ? getExperimentPrimaryMetric(selectedExp) : null
	);
	let selectedCreateExperimentBase = $derived(
		createExperimentBaseExperimentId
			? allExperimentOptions.find((exp) => exp.id === Number(createExperimentBaseExperimentId)) ?? null
			: null
	);
	let selectedEvalMetricRows = $derived.by(() => {
		if (!selectedExp) return [];
		return activeMetricColumns
			.map((column) => ({
				key: column.key,
				label: column.title,
				value: column.render(column.value(selectedExp))
			}))
			.filter((row) => row.value !== '--');
	});
	let resultsScopeLabel = $derived.by(() => {
		switch (resultsScope) {
			case 'adHoc':
				return 'ad hoc runs';
			case 'reruns':
				return 'benchmark reruns';
			case 'benchmark':
				return selectedBenchmarkGroup ? `benchmark ${selectedBenchmarkGroup}` : 'benchmark groups';
			case 'family':
				return selectedModelFamily ? `family ${selectedModelFamily}` : 'model family';
			case 'all':
			default:
				return 'all runs';
		}
	});

	let pastLossRuns = $derived(
		experiments
			.filter(e => e.lossCurve && e.lossCurve.length > 1)
			.map(e => ({
				data: e.lossCurve!,
				color: e.id === selectedExpId ? '#ef4444' : e.kept ? '#22c55e' : '#4b5563',
				highlight: e.id === selectedExpId
			}))
	);

	let trainLoader: DataLoader | null = null;
	let valLoader: DataLoader | null = null;
	let controller: ResearchController | null = null;
	let selectedResearchProfile = $derived(
		selectedResearchProfileId
			? researchProfiles.find((profile) => profile.id === selectedResearchProfileId) ?? null
			: null
	);
	let readyResearchProfile = $derived(
		isConfiguredProfile(selectedResearchProfile) ? selectedResearchProfile : null
	);
	let configuredProfileCount = $derived(
		researchProfiles.filter((profile) => isConfiguredProfile(profile)).length
	);
	let activeDataset = $derived(
		activeDatasetVersionId != null
			? datasetVersions.find((version) => version.id === activeDatasetVersionId) ?? null
			: null
	);
	let currentWorkspace = $derived(
		currentWorkspaceId != null
			? workspaces.find((workspace) => workspace.id === currentWorkspaceId) ?? null
			: null
	);
	let currentWorkspaceBaseExperiment = $derived(
		currentWorkspace?.base_experiment_id != null
			? allExperimentOptions.find((exp) => exp.id === currentWorkspace.base_experiment_id) ?? null
			: null
	);
	let activeTrainer = $derived(
		getTrainerDefinition(activeDataset?.trainerKey ?? DEFAULT_TRAINER_KEY)
	);
	let defaultBaselineCode = $derived(
		getBaselineCodeForTrainer(activeDataset?.trainerKey ?? DEFAULT_TRAINER_KEY)
	);
	let backendSummary = $derived.by(() => {
		if (readyResearchProfile) {
			return `${readyResearchProfile.name || readyResearchProfile.model} ready`;
		}
		if (configuredProfileCount > 0) {
			return `${configuredProfileCount} configured profiles`;
		}
		return 'no configured backend';
	});
	let activeResearchLabel = $derived.by(() => {
		if (!readyResearchProfile) return 'research backend';
		const name = readyResearchProfile.name.trim();
		if (name) return name;
		const model = readyResearchProfile.model.trim();
		if (model) return model;
		return readyResearchProfile.provider === 'openai' ? 'OpenAI-compatible backend' : 'Anthropic-compatible backend';
	});
	let activeDatasetSamplePrompt = $derived(activeDataset?.samplePrompt?.trim() ?? '');
	let canFocusChart = $derived(Boolean(selectedExpId) || lossData.length >= 2);
	let canTrain = $derived(
		Boolean(trainLoader && valLoader && activeDatasetVersionId != null && currentWorkspaceId != null)
	);
	let chartYMin = $derived.by(() => {
		const trimmed = String(chartYMinInput ?? '').trim();
		if (trimmed === '') return null;
		const value = Number(trimmed);
		return Number.isFinite(value) ? value : null;
	});
	let chartYMax = $derived.by(() => {
		const trimmed = String(chartYMaxInput ?? '').trim();
		if (trimmed === '') return null;
		const value = Number(trimmed);
		return Number.isFinite(value) ? value : null;
	});
	let chartManualRangeValid = $derived(
		chartScaleMode !== 'manual' ||
		(chartYMin != null && chartYMax != null && chartYMax > chartYMin)
	);
	let displayStatus = $derived.by(() => {
		if (!running) return status;
		switch (stopIntent) {
			case 'graceful':
				return `${status} · ${waitingForRecommendation ? 'stopping before the next run' : 'stopping after current run'}`;
			case 'immediate':
				return `${status} · stopping immediately`;
			case 'none':
			default:
				return status;
		}
	});
	let stopButtonLabel = $derived.by(() => {
		if (!running) return '';
		if (stopIntent === 'graceful') return 'stop immediately';
		if (stopIntent === 'immediate') return 'stopping...';
		if (waitingForRecommendation) return 'stop after run';
		return mode === 'manual' ? 'stop after run' : 'stop after run';
	});

	onMount(async () => {
		try {
			status = 'initializing';
			await getDb();
			researchProfiles = loadResearchProfiles();
			selectedResearchProfileId = loadActiveResearchProfileId(researchProfiles);
			gpuStatus = await initWebGPU();
			if (!gpuStatus.ok) { status = 'error'; return; }
			await refreshDatasetCatalog();
			await refreshWorkspaces();
			if (currentWorkspaceId != null) {
				await selectWorkspace(currentWorkspaceId);
			} else {
				await loadFromDb();
			}
			const params = new URL(window.location.href).searchParams;
			const expParam = params.get('exp');
			if (expParam) {
				const id = Number(expParam);
				const exp = experiments.find(e => e.id === id);
				if (exp) selectExperiment(exp);
			}
			status = currentWorkspaceId == null
				? 'create an experiment to start training'
				: activeDatasetVersionId == null
					? 'import a dataset to start training'
					: 'ready';
		} catch (e) {
			console.error('Init failed:', e);
			status = 'error';
		}
	});

	$effect(() => {
		if (typeof window === 'undefined') return;
		const datasetId = datasetImportId.trim();
		if (!datasetId) {
			datasetImportInfo = null;
			datasetImportInfoError = '';
			datasetImportInfoBusy = false;
			datasetRecipeDraft = null;
			datasetProposalError = '';
			return;
		}

		let cancelled = false;
		datasetImportInfoBusy = true;
		datasetImportInfoError = '';
		const timeout = window.setTimeout(async () => {
			try {
				const info = await inspectHuggingFaceDataset(datasetId);
				if (cancelled || datasetImportId.trim() !== datasetId) return;
				datasetImportInfo = info;
				applyDatasetImportDefaults(datasetId, info);
				datasetRecipeDraft = { ...info.defaultProposal };
				datasetProposalError = '';
				datasetInspectorOpen = true;
			} catch (error) {
				if (cancelled || datasetImportId.trim() !== datasetId) return;
				datasetImportInfo = null;
				datasetRecipeDraft = null;
				datasetImportInfoError = error instanceof Error ? error.message : String(error);
			} finally {
				if (!cancelled && datasetImportId.trim() === datasetId) {
					datasetImportInfoBusy = false;
				}
			}
		}, 300);

		return () => {
			cancelled = true;
			window.clearTimeout(timeout);
		};
	});

	$effect(() => {
		if (typeof window === 'undefined' || researchProfiles.length === 0) return;
		saveResearchProfiles(researchProfiles);
		saveActiveResearchProfileId(selectedResearchProfileId);
	});

	$effect(() => {
		if (!datasetRecipeDraft) return;
		datasetImportMaxTrain = datasetRecipeDraft.maxTrainExamples == null ? '-' : String(datasetRecipeDraft.maxTrainExamples);
		datasetImportMaxValidation = datasetRecipeDraft.maxValidationExamples == null ? '-' : String(datasetRecipeDraft.maxValidationExamples);
	});

	$effect(() => {
		if (!datasetRecipeDraft) return;
		const parsed = normalizeDraftCount(datasetImportMaxTrain);
		if (parsed !== datasetRecipeDraft.maxTrainExamples) {
			updateDatasetRecipeDraft({ maxTrainExamples: parsed });
		}
	});

	$effect(() => {
		if (!datasetRecipeDraft) return;
		const parsed = normalizeDraftCount(datasetImportMaxValidation);
		if (parsed !== datasetRecipeDraft.maxValidationExamples) {
			updateDatasetRecipeDraft({ maxValidationExamples: parsed });
		}
	});

	$effect(() => {
		if (chartSeriesMode === 'focus' && !canFocusChart) {
			chartSeriesMode = 'all';
		}
	});

	async function loadFromDb() {
		experiments = currentWorkspaceId == null
			? []
			: await getAllExperimentRecords(null, currentWorkspaceId);
		allExperimentOptions = await getAllExperimentRecords();
		const keptChampion = [...experiments]
			.filter((exp) => exp.kept && !exp.error)
			.sort((a, b) => {
				const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
				const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
				return (
					(Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id) ||
					b.id - a.id
				);
			})[0];
		const best = keptChampion ?? [...experiments]
			.filter((exp) => !exp.error && Number.isFinite(exp.valBpb))
			.sort((a, b) => a.valBpb - b.valBpb || b.id - a.id)[0];
		if (best) {
			code = best.code;
		} else {
			code = defaultBaselineCode;
		}
	}

	async function refreshWorkspaces(selectId?: number | null) {
		workspaces = await listExperimentWorkspaces();
		const preferredId = selectId ?? currentWorkspaceId ?? workspaces[0]?.id ?? null;
		currentWorkspaceId = workspaces.some((workspace) => workspace.id === preferredId) ? preferredId : null;
		exportWorkspaceId = currentWorkspaceId != null ? String(currentWorkspaceId) : '';
	}

	function parseOptionalInteger(value: string): number | null {
		const trimmed = value.trim();
		if (!trimmed || trimmed === '-') return null;
		const parsed = Number(trimmed);
		return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
	}

	function formatOptionalCount(value: number | null): string {
		return value == null ? '-' : value.toLocaleString();
	}

	function datasetInspectorTitle(): string {
		if (!datasetImportInfo) {
			return 'dataset prep';
		}
		const mode = datasetRecipeDraft ? datasetRecipeDraft.recipeKey : 'inspect';
		return `dataset prep · ${mode}`;
	}

	function datasetInspectorSubtitle(): string {
		if (!datasetImportInfo) {
			return 'inspect splits, sample rows, and the preprocessing draft before materializing a dataset version';
		}
		const splitMode = datasetRecipeDraft?.validationStrategy === 'carve-from-train'
			? 'carved val'
			: 'hf val';
		return `${formatOptionalCount(datasetImportInfo.trainExamples)} train · ${formatOptionalCount(datasetImportInfo.validationExamples)} val · ${splitMode}`;
	}

	function ensureDatasetPrepVisible(actionLabel: string): boolean {
		if (currentWorkspaceId == null) {
			setupTab = 'createExperiment';
			status = `create an experiment before ${actionLabel}`;
			return false;
		}
		if (activeDatasetVersionId == null) {
			status = `select a dataset before ${actionLabel}`;
			return false;
		}
		if (
			setupTab === 'experiments' &&
			activeDatasetReviewOpen &&
			datasetPrepConfirmedVersionId === activeDatasetVersionId
		) {
			return true;
		}

		setupTab = 'experiments';
		activeDatasetReviewOpen = true;
		datasetPrepConfirmedVersionId = activeDatasetVersionId;
		status = `review active dataset before ${actionLabel}`;
		return false;
	}

	function normalizeDraftCount(value: string): number | null {
		const trimmed = value.trim();
		if (!trimmed || trimmed === '-') return null;
		const parsed = Number(trimmed);
		return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
	}

	function updateDatasetRecipeDraft(patch: Partial<DatasetRecipeProposal>) {
		if (!datasetRecipeDraft) return;
		datasetRecipeDraft = {
			...datasetRecipeDraft,
			...patch
		};
	}

	function selectedRecipeOption(recipeKey: string) {
		return datasetImportInfo?.recipeOptions.find((option) => option.key === recipeKey) ?? null;
	}

	function applyRecipeOptionToDraft(recipeKey: string) {
		const option = selectedRecipeOption(recipeKey);
		if (!option || !datasetRecipeDraft) return;
		datasetRecipeDraft = {
			...datasetRecipeDraft,
			trainerKey: option.trainerKey,
			modelFamily: option.modelFamily,
			recipeKey: option.key,
			textFields: option.textFields.length > 0 ? option.textFields : datasetRecipeDraft.textFields,
			preprocessingSummary: option.preprocessingSummary,
			preprocessingSteps: option.preprocessingSteps,
			researchNotes: option.researchNotes,
			samplePrompt: option.samplePrompt
		};
	}

	function updateDatasetRecipeDraftCounts(field: 'maxTrainExamples' | 'maxValidationExamples', value: string) {
		updateDatasetRecipeDraft({ [field]: normalizeDraftCount(value) } as Pick<DatasetRecipeProposal, typeof field>);
	}

	function updateDatasetRecipeDraftTextFields(value: string) {
		if (!datasetImportInfo) return;
		const selected = value
			.split(',')
			.map((field) => field.trim())
			.filter((field, index, values) => field.length > 0 && values.indexOf(field) === index)
			.filter((field) => datasetImportInfo.featureNames.includes(field));
		updateDatasetRecipeDraft({
			textFields: selected.length > 0 ? selected : datasetImportInfo.featureNames
		});
	}

	function applyInferencePromptPreset() {
		if (!activeDatasetSamplePrompt || sampling) return;
		prompt = activeDatasetSamplePrompt;
	}

	function availableRecipeDescription(recipeKey: string): string {
		return datasetImportInfo?.recipeOptions.find((option) => option.key === recipeKey)?.description ?? '';
	}

	async function requestDatasetRecipeProposal() {
		if (!datasetImportInfo || datasetProposalBusy) return;
		if (!readyResearchProfile) {
			datasetProposalError = 'configure a research backend before asking for a recipe proposal';
			return;
		}

		datasetProposalBusy = true;
		datasetProposalError = '';
		try {
			const { proposal } = await proposeDatasetRecipe({
				inspection: datasetImportInfo,
				profile: readyResearchProfile
			});
			datasetRecipeDraft = proposal;
		} catch (error) {
			datasetProposalError = error instanceof Error ? error.message : String(error);
		} finally {
			datasetProposalBusy = false;
		}
	}

	function datasetTrainLabel(): string {
		if (!datasetImportInfo || datasetImportInfo.trainExamples == null) {
			return 'train';
		}
		return `train (max: ${formatOptionalCount(datasetImportInfo.trainExamples)})`;
	}

	function datasetValLabel(): string {
		if (!datasetImportInfo) {
			return 'val';
		}
		if (datasetImportInfo.hasValidationSplit) {
			return `val (max: ${formatOptionalCount(datasetImportInfo.validationExamples)})`;
		}
		return 'val (split from train)';
	}

	function applyDatasetImportDefaults(datasetId: string, info: HuggingFaceDatasetInspection) {
		const nextTrain = info.trainExamples != null ? String(info.trainExamples) : '';
		const nextValidation = info.hasValidationSplit
			? (info.validationExamples != null ? String(info.validationExamples) : '')
			: '-';

		const shouldUpdateTrain =
			datasetImportDefaultsSourceId !== datasetId ||
			datasetImportMaxTrain.trim() === '' ||
			datasetImportMaxTrain === datasetImportSuggestedTrain;
		const shouldUpdateValidation =
			datasetImportDefaultsSourceId !== datasetId ||
			datasetImportMaxValidation.trim() === '' ||
			datasetImportMaxValidation === datasetImportSuggestedValidation;

		if (shouldUpdateTrain) {
			datasetImportMaxTrain = nextTrain;
		}
		if (shouldUpdateValidation) {
			datasetImportMaxValidation = nextValidation;
		}

		datasetImportSuggestedTrain = nextTrain;
		datasetImportSuggestedValidation = nextValidation;
		datasetImportDefaultsSourceId = datasetId;
	}

	function datasetImportHelperText(): string {
		if (datasetImportInfoBusy) {
			return 'checking Hugging Face split metadata...';
		}
		if (datasetImportInfoError) {
			return datasetImportInfoError;
		}
		if (!datasetImportInfo) {
			return 'enter a Hugging Face dataset id to inspect train and validation split sizes.';
		}
		if (datasetImportInfo.hasValidationSplit) {
			return `config ${datasetImportInfo.configName} · train ${formatOptionalCount(datasetImportInfo.trainExamples)} · validation ${formatOptionalCount(datasetImportInfo.validationExamples)} · splits ${datasetImportInfo.splitNames.join(', ')}`;
		}
		return `config ${datasetImportInfo.configName} · train ${formatOptionalCount(datasetImportInfo.trainExamples)} · no validation split on Hugging Face · enter a validation count to carve from train`;
	}

	function formatDatasetVersionSummary(version: DatasetVersionSummary): string {
		const datasetNumber = datasetVersionNumberById.get(version.id) ?? version.id;
		if (version.sourceType === 'legacy') {
			return `#${datasetNumber} · ${version.label} · ${Math.round(version.trainBytes / 1024)} KB legacy`;
		}
		return `#${datasetNumber} · ${version.label} · ${version.trainExamples.toLocaleString()} train`;
	}

	function formatWorkspaceSummary(workspace: ExperimentWorkspaceRecord): string {
		const sourceLabel = workspace.source_type === 'dataset' ? 'dataset' : 'results';
		if (workspace.dataset_version_id != null) {
			const dataset = datasetVersions.find((version) => version.id === workspace.dataset_version_id);
			if (dataset) {
				const datasetNumber = datasetVersionNumberById.get(dataset.id) ?? dataset.id;
				return `${workspace.name} · ${sourceLabel} · #${datasetNumber} ${dataset.label}`;
			}
		}
		if (workspace.base_experiment_id != null) {
			const base = allExperimentOptions.find((exp) => exp.id === workspace.base_experiment_id);
			if (base) {
				return `${workspace.name} · results · ${base.name}`;
			}
		}
		return `${workspace.name} · ${sourceLabel}`;
	}

	function formatExperimentOption(exp: ExperimentRecord): string {
		const sourceLabel = exp.source === 'auto' ? 'A' : 'M';
		const datasetLabel = exp.datasetLabel?.trim() || 'unscoped dataset';
		return `${sourceLabel} · #${exp.id} · ${exp.name} · ${datasetLabel}`;
	}

	async function loadDatasetLoaders(versionId: number | null) {
		if (versionId == null) {
			trainLoader = null;
			valLoader = null;
			status = 'import a dataset to start training';
			return;
		}

		status = 'loading dataset...';
		[trainLoader, valLoader] = await Promise.all([
			DataLoader.fetch(`/api/datasets/${versionId}/train`),
			DataLoader.fetch(`/api/datasets/${versionId}/val`)
		]);
	}

	async function refreshDatasetCatalog(catalog?: { versions: DatasetVersionSummary[]; activeVersionId: number | null }) {
		const nextCatalog = catalog ?? await listDatasetVersions();
		datasetVersions = nextCatalog.versions;
		activeDatasetVersionId = nextCatalog.activeVersionId;
		datasetPrepConfirmedVersionId = null;
		await loadDatasetLoaders(activeDatasetVersionId);
	}

	async function importHuggingFaceDataset() {
		if (running || importing || datasetBusy) return;

		datasetBusy = true;
		try {
			status = `importing ${datasetImportId.trim() || 'dataset'}...`;
			const importDraft = datasetRecipeDraft;
			const catalog = await importDatasetFromHuggingFace({
				datasetId: datasetImportId,
				label: importDraft?.label?.trim() || undefined,
				recipeKey: importDraft?.recipeKey ?? undefined,
				textFields: importDraft?.textFields ?? undefined,
				samplePrompt: importDraft?.samplePrompt?.trim() || undefined,
				maxTrainExamples: importDraft?.maxTrainExamples ?? parseOptionalInteger(datasetImportMaxTrain),
				maxValidationExamples: importDraft?.maxValidationExamples ?? parseOptionalInteger(datasetImportMaxValidation)
			});
			applyResultsScope('all');
			setSelectedExp(null);
			loadedModel = null;
			await refreshDatasetCatalog(catalog);
			await loadFromDb();
			datasetInspectorOpen = false;
			status = 'ready';
		} catch (e) {
			console.error('Dataset import failed:', e);
			status = `dataset import failed: ${e}`;
		} finally {
			datasetBusy = false;
		}
	}

	async function activateDatasetVersion(versionId: number) {
		if (running || importing || datasetBusy || versionId === activeDatasetVersionId) return;

		datasetBusy = true;
		try {
			setSelectedExp(null);
			applyResultsScope('all');
			loadedModel = null;
			await refreshDatasetCatalog(await setActiveDatasetVersion(versionId));
			await loadFromDb();
			datasetInspectorOpen = false;
			status = 'ready';
		} catch (e) {
			console.error('Failed to activate dataset:', e);
			status = `dataset switch failed: ${e}`;
		} finally {
			datasetBusy = false;
		}
	}

	async function selectWorkspace(workspaceId: number) {
		if (running || importing || datasetBusy || workspaceId === currentWorkspaceId) return;
		currentWorkspaceId = workspaceId;
		exportWorkspaceId = String(workspaceId);
		const workspace = workspaces.find((entry) => entry.id === workspaceId) ?? null;
		if (workspace?.dataset_version_id != null && workspace.dataset_version_id !== activeDatasetVersionId) {
			await activateDatasetVersion(workspace.dataset_version_id);
		} else {
			await loadFromDb();
		}
		if (workspace?.base_experiment_id != null) {
			const base = allExperimentOptions.find((exp) => exp.id === workspace.base_experiment_id) ?? null;
			if (base) {
				code = base.code;
			}
		}
		setupTab = 'experiments';
		status = workspace ? `workspace ready: ${workspace.name}` : 'ready';
	}

	async function handleCreateWorkspace() {
		if (createExperimentBusy) return;
		const name = createExperimentName.trim();
		if (!name) {
			status = 'name the experiment workspace before creating it';
			return;
		}
		const datasetVersionId = createExperimentSourceType === 'dataset'
			? Number(createExperimentDatasetId || activeDatasetVersionId || 0) || null
			: selectedCreateExperimentBase?.datasetVersionId ?? null;
		const baseExperimentId = createExperimentSourceType === 'results'
			? Number(createExperimentBaseExperimentId || 0) || null
			: null;
		if (createExperimentSourceType === 'dataset' && datasetVersionId == null) {
			status = 'pick a dataset-backed source for this experiment workspace';
			return;
		}
		if (createExperimentSourceType === 'results' && baseExperimentId == null) {
			status = 'pick a results-backed source experiment for this workspace';
			return;
		}
		createExperimentBusy = true;
		try {
			const workspaceId = await createExperimentWorkspace({
				name,
				sourceType: createExperimentSourceType,
				datasetVersionId,
				baseExperimentId,
				readme: createExperimentReadme,
				notes: createExperimentNotes
			});
			await refreshWorkspaces(workspaceId);
			await selectWorkspace(workspaceId);
			createExperimentName = '';
			createExperimentReadme = '';
			createExperimentNotes = '';
			setupTab = 'experiments';
		} finally {
			createExperimentBusy = false;
		}
	}

	function makeBenchmarkGroupLabel() {
		return new Date().toISOString().replace('T', ' ').slice(0, 19);
	}

	function clearBatchSelection() {
		selectedBatchIds = [];
	}

	function applyResultsScope(scope: ResultsScope, value?: string) {
		resultsScope = scope;
		selectedBenchmarkGroup = scope === 'benchmark' ? value ?? null : null;
		selectedModelFamily = scope === 'family' ? value ?? null : null;
		selectionMode = false;
		clearBatchSelection();
	}

	function toggleBatchSelection(expId: number) {
		selectedBatchIds = selectedBatchIds.includes(expId)
			? selectedBatchIds.filter((id) => id !== expId)
			: [...selectedBatchIds, expId];
	}

	function resetStopIntent() {
		stopIntent = 'none';
	}

	function stopCurrentRunImmediately() {
		trainAbort?.abort();
		controller?.stopImmediately();
		controller?.stopCurrentRun();
	}

	function requestRunStop() {
		if (!running) return;
		if (stopIntent === 'immediate') return;
		if (stopIntent === 'graceful') {
			stopIntent = 'immediate';
			stopCurrentRunImmediately();
			return;
		}

		stopIntent = 'graceful';
		if (!controller) return;
		if (waitingForRecommendation) {
			controller.stopImmediately();
			return;
		}
		controller.requestStopAfterCurrentRun();
	}

	type LocalRunRequest = {
		name: string;
		source: 'manual' | 'auto';
		code: string;
		reasoning: string;
		rerunOf?: number | null;
		benchmarkGroup?: string | null;
		saveWeights?: boolean;
		generateSample?: boolean;
		timeoutMs?: number;
	};

	type LocalRunOutcome = {
		record: ExperimentRecord;
		result: RunResult;
		dbId: number;
	};

	async function runLocalExperiment(request: LocalRunRequest): Promise<LocalRunOutcome> {
		if (!trainLoader || !valLoader || !activeDataset) {
			throw new Error('training data is not ready');
		}

		lossData = [];
		inferences = [];
		inferenceIdx = 0;
		currentRunName = request.name;
		currentReasoning = request.reasoning;
		status = request.rerunOf
			? `rerunning #${request.rerunOf}...`
			: 'training...';
		trainAbort = new AbortController();

		inProgressExp = {
			id: -1,
			name: request.name,
			source: request.source,
			code: request.code,
			datasetVersionId: activeDataset.id,
			datasetLabel: activeDataset.label,
			datasetSourceRef: activeDataset.sourceRef,
			trainerKey: activeDataset.trainerKey,
			modelFamily: activeDataset.modelFamily,
			valBpb: Infinity,
			elapsed: 0,
			totalSteps: 0,
			reasoning: request.reasoning,
			kept: false,
			rerunOf: request.rerunOf ?? null,
			benchmarkGroup: request.benchmarkGroup ?? null
		};

		const result = await executeTrainCode(request.code, trainLoader, valLoader, 30, activeDataset.trainerKey, {
			signal: trainAbort.signal,
			timeoutMs: request.timeoutMs,
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				status = `step ${m.step} | loss ${m.loss.toFixed(4)} | ${(m.elapsed / 1000).toFixed(1)}s`;
				if (inProgressExp) {
					inProgressExp = {
						...inProgressExp,
						valBpb: m.loss,
						totalSteps: m.step,
						elapsed: m.elapsed
					};
				}
			}
		});
		trainAbort = null;
		inProgressExp = null;

		const kept = experiments.length === 0 || result.valBpb < Math.min(...experiments.map((e) => e.valBpb));
		const dbId = await insertExperiment({
			projectId: currentWorkspaceId,
			name: request.name,
			source: request.source,
			code: request.code,
			datasetVersionId: activeDataset.id,
			datasetLabel: activeDataset.label,
			datasetSourceRef: activeDataset.sourceRef,
			trainerKey: activeDataset.trainerKey,
			modelFamily: activeDataset.modelFamily,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning: request.reasoning,
			kept,
			lossCurve: lossData,
			error: result.error,
			rerunOf: request.rerunOf ?? null,
			benchmarkGroup: request.benchmarkGroup ?? null
		});

		await insertLossCurve(dbId, lossData);

		if (!result.error) {
			loadedModel = {
				forward: result.forward,
				params: result.params,
				vocabSize: result.vocabSize,
				seqLen: result.seqLen,
				expId: dbId
			};
		}

		if (request.saveWeights && result.params && Object.keys(result.params).length > 0) {
			(async () => {
				try {
					const weightsPath = await saveWeights(dbId, result.params);
					await updateWeightsPath(dbId, weightsPath);
				} catch (e) {
					console.error('Failed to save weights:', e);
				}

				if (request.generateSample && loadedModel?.forward && loadedModel.expId === dbId) {
					try {
						const output = await sampleText(
							result.params,
							loadedModel.forward,
							result.vocabSize,
							result.seqLen,
							'',
							200,
							0.8
						);
						await insertInference({ experimentId: dbId, prompt: '', output, temperature: 0.8 });
						if (selectedExpId === dbId) {
							inferences = await getInferencesForExperiment(dbId);
							inferenceIdx = 0;
						}
					} catch (e) {
						console.error('Failed to generate sample:', e);
					}
				}
			})();
		}

		return {
			dbId,
			result,
			record: {
				id: dbId,
				name: request.name,
				source: request.source,
				code: request.code,
				datasetVersionId: activeDataset.id,
				datasetLabel: activeDataset.label,
				datasetSourceRef: activeDataset.sourceRef,
				trainerKey: activeDataset.trainerKey,
				modelFamily: activeDataset.modelFamily,
				valBpb: result.valBpb,
				elapsed: result.elapsed,
				totalSteps: result.totalSteps,
				reasoning: request.reasoning,
				kept,
				error: result.error,
				lossCurve: lossData,
				rerunOf: request.rerunOf ?? null,
				benchmarkGroup: request.benchmarkGroup ?? null
			}
		};
	}

	async function startManualTraining() {
		if (!canTrain || running) return;
		if (!ensureDatasetPrepVisible('running training')) return;

		running = true;
		resetStopIntent();
		datasetInspectorOpen = false;
		setListMode('current');

		try {
			const runName = experimentName || petname();
			const outcome = await runLocalExperiment({
				name: runName,
				source: 'manual',
				code,
				reasoning: 'Manual run',
				saveWeights: true,
				generateSample: true
			});
			await loadFromDb();
			selectExperimentById(outcome.dbId);
			const wasStoppedImmediately = stopIntent === 'immediate';
			const wasStoppedGracefully = stopIntent === 'graceful';
			status = wasStoppedImmediately && outcome.result.error?.toLowerCase().includes('aborted')
				? 'stopped immediately'
				: wasStoppedGracefully
					? 'stopped after current run'
					: outcome.result.error
						? `error: ${outcome.result.error}`
						: `done — ${activeTrainer.metricKey}: ${outcome.result.valBpb.toFixed(4)} | ${outcome.result.totalSteps} steps`;
			experimentName = petname();
		} finally {
			running = false;
			resetStopIntent();
		}
	}

	async function rerunExperiments(targets: ExperimentRecord[]) {
		if (!trainLoader || !valLoader || running || targets.length === 0) return;

		running = true;
		resetStopIntent();
		setListMode('current');
		const benchmarkGroup = makeBenchmarkGroupLabel();
		const saveWeights = targets.length === 1;
		let completed = 0;
		let failures = 0;
		let lastDbId: number | null = null;

		try {
			for (const exp of targets) {
				const outcome = await runLocalExperiment({
					name: exp.name,
					source: exp.source,
					code: exp.code,
					reasoning: `Benchmark rerun of #${exp.id}${exp.reasoning ? ` — ${exp.reasoning}` : ''}`,
					rerunOf: exp.id,
					benchmarkGroup,
					saveWeights,
					generateSample: false,
					timeoutMs: 180000
				});
				completed += 1;
				if (outcome.result.error) failures += 1;
				lastDbId = outcome.dbId;
				await loadFromDb();

				if (outcome.result.error?.toLowerCase().includes('aborted')) {
					status = stopIntent === 'immediate'
						? `stopped immediately after ${completed}/${targets.length} reruns in ${benchmarkGroup}`
						: `stopped after ${completed}/${targets.length} reruns in ${benchmarkGroup}`;
					return;
				}

				if (stopIntent === 'graceful') {
					status = `stopped after ${completed}/${targets.length} reruns in ${benchmarkGroup}`;
					return;
				}
			}

			if (lastDbId != null) {
				selectExperimentById(lastDbId);
			}
			status = failures > 0
				? `reran ${completed} experiments in baseline ${benchmarkGroup} (${failures} failed)`
				: `reran ${completed} experiment${completed === 1 ? '' : 's'} in baseline ${benchmarkGroup}`;
		} finally {
			running = false;
			resetStopIntent();
			clearBatchSelection();
			selectionMode = false;
		}
	}

	async function rerunSelectedExperiment() {
		if (!selectedExp) return;
		await rerunExperiments([selectedExp]);
	}

	async function rerunSelectedBatch() {
		if (selectedBatchExperiments.length === 0) return;
		await rerunExperiments(selectedBatchExperiments);
	}

	async function rerunAllExperiments() {
		const originals = experiments.filter((exp) => !exp.rerunOf);
		if (originals.length === 0) return;
		await rerunExperiments(originals);
	}

	async function deleteExperimentRecords(ids: number[], scopeLabel: string) {
		if (ids.length === 0 || running || importing) return;

		const confirmed = window.confirm(
			`Delete ${scopeLabel}? This removes the experiment rows, their loss curves, inferences, and any saved weights for those runs.`
		);
		if (!confirmed) return;

		const selectedIdBeforeDelete = selectedExpId;
		const deletedCount = await deleteExperiments(ids);
		await loadFromDb();

		selectedBatchIds = selectedBatchIds.filter((id) => !ids.includes(id));
		if (selectedIdBeforeDelete != null && ids.includes(selectedIdBeforeDelete)) {
			setSelectedExp(null);
			code = experiments.length > 0 ? code : defaultBaselineCode;
		}

		status = `deleted ${deletedCount} experiment${deletedCount === 1 ? '' : 's'}`;
	}

	async function deleteSelectedExperiment() {
		if (!selectedExp) return;
		await deleteExperimentRecords([selectedExp.id], `experiment #${selectedExp.id}`);
	}

	async function deleteSelectedBatch() {
		if (selectedBatchIds.length === 0) return;
		await deleteExperimentRecords(selectedBatchIds, `${selectedBatchIds.length} selected experiments`);
	}

	async function startResearch() {
		if (!trainLoader || !valLoader || !activeDataset || running) return;
		if (!readyResearchProfile) {
			status = 'configure a research backend before starting';
			return;
		}
		if (!ensureDatasetPrepVisible('starting research')) return;

		running = true;
		resetStopIntent();
		waitingForRecommendation = true;
		lossData = [];
		status = 'starting...';
		datasetInspectorOpen = false;
		controller = new ResearchController();
		controller.profile = readyResearchProfile;
		controller.datasetContext = {
			versionId: activeDataset.id,
			label: activeDataset.label,
			sourceRef: activeDataset.sourceRef,
			recipeKey: activeDataset.recipeKey,
			recipeDescription: activeDataset.recipeDescription,
			preprocessingSummary: activeDataset.preprocessingSummary,
			preprocessingSteps: activeDataset.preprocessingSteps,
			researchNotes: activeDataset.researchNotes,
			samplePrompt: activeDataset.samplePrompt,
			trainerKey: activeDataset.trainerKey,
			modelFamily: activeDataset.modelFamily,
			vocabSize: activeDataset.vocabSize,
			trainBytes: activeDataset.trainBytes,
			validationBytes: activeDataset.validationBytes,
			textFields: activeDataset.textFields
		};
		controller.bestCode = getBaselineCodeForTrainer(activeDataset.trainerKey);
		setListMode('current');

		controller.history = [...experiments];

		let requestedStop: 'none' | 'graceful' | 'immediate' = 'none';
		try {
			await controller.run(trainLoader, valLoader, {
				onExperimentStart(expCode, reasoning) {
					waitingForRecommendation = false;
					lossData = [];
					code = expCode;
					currentReasoning = reasoning;
					currentRunName = petname();
					status = `experiment: ${reasoning}`;
					if (listMode === 'current') setSelectedExp(null);
					inProgressExp = {
						id: -1, name: currentRunName, source: 'auto', code: expCode,
						datasetVersionId: activeDataset?.id ?? null,
						datasetLabel: activeDataset?.label ?? null,
						datasetSourceRef: activeDataset?.sourceRef ?? null,
						trainerKey: activeDataset?.trainerKey ?? DEFAULT_TRAINER_KEY,
						modelFamily: activeDataset?.modelFamily ?? 'byte-gpt',
						valBpb: Infinity, elapsed: 0, totalSteps: 0, reasoning, kept: false,
					};
				},
				onStep(m: StepMetrics) {
					lossData = [...lossData, { step: m.step, loss: m.loss }];
					if (inProgressExp) {
						inProgressExp = { ...inProgressExp, valBpb: m.loss, totalSteps: m.step, elapsed: m.elapsed };
					}
				},
				async onExperimentDone(record: ExperimentRecord) {
					inProgressExp = null;
					waitingForRecommendation = true;
					await loadFromDb();
					if (listMode === 'current') setSelectedExp(null);
					if (record.kept && controller) {
						code = controller.bestCode;
					}
					status = `#${record.id} ${record.kept ? 'KEPT' : 'discarded'} — ${activeTrainer.metricShortLabel} ${record.valBpb.toFixed(4)}`;
				},
				onCodeStream(streamedCode) {
					code = streamedCode;
				},
				onReasoningStream(streamedReasoning) {
					currentReasoning = streamedReasoning;
					status = `thinking: ${streamedReasoning}`;
				},
				onError(error) {
					status = `error: ${error}`;
				}
			});

			requestedStop = stopIntent;
			if (requestedStop === 'graceful') {
				status = 'stopped after current run';
			} else if (requestedStop === 'immediate') {
				status = 'stopped immediately';
			}
		} catch (error) {
			status = `error: ${error instanceof Error ? error.message : String(error)}`;
		} finally {
			inProgressExp = null;
			waitingForRecommendation = false;
			running = false;
			resetStopIntent();
		}
	}

	function stopCurrentRun() {
		requestRunStop();
	}

	function setSelectedExp(id: number | null) {
		const normalizedId = id != null && Number.isInteger(id) && id > 0 ? id : null;
		selectedExpId = normalizedId;
		inferences = [];
		inferenceIdx = 0;
		streamingOutput = '';
		sampling = false;
		const url = new URL(window.location.href);
		if (normalizedId != null) url.searchParams.set('exp', String(normalizedId));
		else url.searchParams.delete('exp');
		history.replaceState(null, '', url);
		if (normalizedId != null) {
			getInferencesForExperiment(normalizedId).then(rows => {
				if (selectedExpId === normalizedId) inferences = rows;
			});
		}
	}

	function setListMode(m: 'leaderboard' | 'current') {
		listMode = m;
		leaderboardSort = m === 'leaderboard' ? 'bpb' : 'newest';
		if (m === 'current') setSelectedExp(null);
	}

	function selectExperimentById(id: number) {
		setSelectedExp(id);
	}

	function selectExperiment(exp: ExperimentRecord) {
		code = exp.code;
		selectExperimentById(exp.id);
	}

	async function loadModelForExperiment(expId: number): Promise<boolean> {
		const exp = experiments.find(e => e.id === expId);
		if (!exp) return false;

		// If we already have this model loaded, skip
		if (loadedModel && loadedModel.expId === expId) return true;

		// Load saved weights
		const savedParams = await loadWeights(expId);
		if (!savedParams) return false;

		// Re-execute the code with trainSeconds=0 to get the forward function
		if (!trainLoader || !valLoader) return false;
		const result = await executeTrainCode(exp.code, trainLoader, valLoader, 0, exp.trainerKey ?? DEFAULT_TRAINER_KEY, {
			signal: new AbortController().signal,
			onStep() {},
		});

		if (!result.forward || result.error) return false;

		// Use the saved weights (not the freshly-initialized ones)
		loadedModel = {
			forward: result.forward,
			params: savedParams,
			vocabSize: result.vocabSize,
			seqLen: result.seqLen,
			expId,
		};
		return true;
	}

	async function generateSample() {
		if (running || sampling || !selectedExpId) return;
		const experimentId = selectedExpId;
		const promptText = prompt;
		const sampleTemperature = temperature;
		sampling = true;
		try {
			if (!loadedModel || loadedModel.expId !== experimentId) {
				status = 'loading model...';
				const loaded = await loadModelForExperiment(experimentId);
				if (!loaded) {
					console.error('Could not load model for experiment', experimentId);
					sampling = false;
					status = 'ready';
					return;
				}
				status = 'ready';
			}
			streamingOutput = '';
			const output = await sampleText(loadedModel!.params, loadedModel!.forward, loadedModel!.vocabSize, loadedModel!.seqLen, promptText, 200, sampleTemperature, (text) => {
				streamingOutput = text;
			});
			await insertInference({ experimentId, prompt: promptText, output, temperature: sampleTemperature });
			streamingOutput = '';
			const rows = await getInferencesForExperiment(experimentId);
			if (selectedExpId === experimentId) {
				inferences = rows;
				inferenceIdx = 0;
			}
		} catch (e) {
			console.error('Inference failed:', e);
		}
		sampling = false;
	}

	let showClearModal = $state(false);

	function handleClear() { showClearModal = true; }

	async function confirmClear() {
		showClearModal = false;
		await clearAllData();
		experiments = [];
		code = defaultBaselineCode;
		setSelectedExp(null);
		applyResultsScope('all');
		inferences = [];
		loadedModel = null;
		status = activeDataset ? 'ready' : 'import a dataset to start training';
	}

	async function handleExport() {
		const projectId = exportWorkspaceId ? Number(exportWorkspaceId) : currentWorkspaceId;
		const blob = await exportCsvZip(projectId ?? undefined);
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = projectId ? `autoresearch-project-${projectId}.zip` : 'autoresearch-experiments.zip';
		a.click();
		URL.revokeObjectURL(url);
	}

	function handleImportClick() {
		if (running || importing) return;
		importInput?.click();
	}

	async function handleImportFile(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		importing = true;
		status = 'importing experiments...';

		try {
			const summary = await importCsvZip(file);
			await refreshWorkspaces(currentWorkspaceId);
			if (currentWorkspaceId != null) {
				await selectWorkspace(currentWorkspaceId);
			} else {
				await loadFromDb();
			}
			status = `imported ${summary.addedWorkspaces} workspaces, ${summary.addedExperiments} experiments, ${summary.addedLossSteps} loss steps, ${summary.addedInferences} inferences, ${summary.addedWeights} weights`;
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			status = `import failed: ${message}`;
		} finally {
			importing = false;
			input.value = '';
		}
	}

	let currentInference = $derived(inferences.length > 0 ? inferences[inferenceIdx] : null);
	let selectedExp = $derived(selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null);
	let activeDatasetPrepReady = $derived(
		activeDatasetVersionId != null &&
		datasetPrepConfirmedVersionId === activeDatasetVersionId &&
		setupTab === 'experiments' &&
		activeDatasetReviewOpen
	);
	let datasetVersionNumberById = $derived.by(() => {
		const sorted = [...datasetVersions].sort((a, b) => {
			const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
			const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
			return (Number.isFinite(aTime) ? aTime : a.id) - (Number.isFinite(bTime) ? bTime : b.id) || a.id - b.id;
		});
		return new Map(sorted.map((version, index) => [version.id, index + 1]));
	});
	let datasetRunNumberById = $derived.by(() => {
		const sorted = [...experiments].sort((a, b) => {
			const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
			const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
			return (Number.isFinite(aTime) ? aTime : a.id) - (Number.isFinite(bTime) ? bTime : b.id) || a.id - b.id;
		});
		return new Map(sorted.map((exp, index) => [exp.id, index + 1]));
	});
	let nextDatasetRunNumber = $derived(experiments.length + 1);
	let selectedRunNumber = $derived(selectedExp ? datasetRunNumberById.get(selectedExp.id) ?? null : null);
	let hasAnyModel = $derived(experiments.length > 0 || running);
	let isFirstLoad = $derived(false);
</script>

<svelte:head>
	<title>autoresearch, in the browser!</title>
	<meta name="description" content="Train small language models in your browser using WebGPU. Claude writes the training code, runs experiments, and iterates — a model training another model!" />
	<meta property="og:title" content="autoresearch, in the browser!" />
	<meta property="og:description" content="Train small language models in your browser using WebGPU. Claude writes the training code, runs experiments, and iterates." />
	<meta property="og:image" content="https://autoresearch.lucasgelfond.online/demo.gif" />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="autoresearch, in the browser!" />
	<meta name="twitter:description" content="Train small language models in your browser using WebGPU. Claude writes the training code and iterates." />
	<meta name="twitter:image" content="https://autoresearch.lucasgelfond.online/demo.gif" />
</svelte:head>

<main class="mx-auto w-full max-w-[112rem] space-y-5 px-4 py-8 lg:px-6 lg:py-10 2xl:max-w-[124rem]">
	<!-- Mobile warning -->
	<div class="md:hidden rounded border-2 border-red-600 bg-red-950 p-4 font-mono text-sm text-red-300 text-center leading-relaxed">
		<span class="font-bold text-red-400">WARNING:</span> mobile devices often crash with memory intensive web activities. best results on a proper computer!
	</div>

	{#if gpuStatus && !gpuStatus.ok}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400">
			{gpuStatus.reason}
		</div>
	{:else}
		<div class="max-w-xl space-y-3">
			<h1 class="text-lg font-mono font-bold text-white">autoresearch, in the browser!</h1>
			<p class="text-xs font-mono text-gray-300 leading-relaxed">
				Train a small language model (all on device, with WebGPU!) by generating training code, running experiments, and iterating on progress.
			</p>
			<p class="text-[10px] font-mono text-gray-500 leading-relaxed">
				Based on Andrej Karpathy's <a href="https://github.com/karpathy/autoresearch" class="underline hover:text-gray-200">autoresearch</a> and built on <a href="https://www.ekzhang.com/" class="underline hover:text-gray-200">Eric Zhang</a>'s <a href="https://github.com/ekzhang/jax-js" class="underline hover:text-gray-200">jax-js</a>. Built by <a href="https://lucasgelfond.online" class="underline hover:text-gray-200">Lucas Gelfond</a>. Source <a href="https://github.com/lucasgelfond/autoresearch-webgpu" class="underline hover:text-gray-200">here</a>.
			</p>
		</div>
		{#if status === 'error'}
			<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400 mt-4">
				something went wrong during initialization. check the console for details.
			</div>
		{:else if isFirstLoad}
		<div class="space-y-4">
			<div class="rounded border border-gray-800 bg-gray-950/60 overflow-hidden">
				<div class="border-b border-gray-800 px-3 pt-3">
					<div class="flex items-center gap-2">
						<button
							type="button"
							onclick={() => (setupTab = 'experiments')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'experiments' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>experiments</button>
						<button
							type="button"
							onclick={() => (setupTab = 'createExperiment')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'createExperiment' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>create experiment</button>
						<button
							type="button"
							onclick={() => (setupTab = 'importDataset')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'importDataset' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>import dataset</button>
						<button
							type="button"
							onclick={() => (setupTab = 'importResults')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'importResults' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>import results</button>
						<button
							type="button"
							onclick={() => (setupTab = 'exportResults')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'exportResults' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>export results</button>
						<button
							type="button"
							onclick={() => (setupTab = 'research')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'research' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>research backends</button>
					</div>
				</div>
				<div class="p-4">
					{#if setupTab === 'datasets'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">datasets</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Import a Hugging Face dataset, create local train and validation artifacts for its assigned trainer, then point the browser runner at that version.
								</p>
							</div>
							<div class="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_110px_110px_auto] md:items-end">
								<label class="block">
									<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">dataset id</span>
									<input
										type="text"
										bind:value={datasetImportId}
										placeholder="huggingface dataset id"
										disabled={running || importing || datasetBusy}
										class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
									/>
								</label>
								<label class="block">
									<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">{datasetTrainLabel()}</span>
									<input
										type="text"
										inputmode="numeric"
										bind:value={datasetImportMaxTrain}
										disabled={running || importing || datasetBusy}
										placeholder={datasetImportSuggestedTrain || 'train rows'}
										class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
										title="max train examples"
									/>
								</label>
								<label class="block">
									<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">{datasetValLabel()}</span>
									<input
										type="text"
										inputmode="numeric"
										bind:value={datasetImportMaxValidation}
										disabled={running || importing || datasetBusy}
										placeholder={datasetImportSuggestedValidation || '-'}
										class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
										title="max validation examples"
									/>
								</label>
								<button
									type="button"
									onclick={importHuggingFaceDataset}
									disabled={running || importing || datasetBusy}
									class="rounded border border-emerald-800 px-3 py-1.5 font-mono text-xs text-emerald-300 hover:border-emerald-600 hover:text-emerald-200 disabled:opacity-40 transition-colors"
								>
									{datasetBusy ? 'syncing...' : 'import hf'}
								</button>
							</div>
							<p class="text-[10px] font-mono {datasetImportInfoError ? 'text-amber-300' : 'text-gray-500'}">
								{datasetImportHelperText()}
							</p>
							<div class="flex items-center justify-between gap-3 rounded border border-gray-800 bg-black/20 px-3 py-2">
								<p class="text-[10px] font-mono text-gray-500">
									Restore experiment history from a prior export ZIP without leaving dataset setup.
								</p>
								<button
									type="button"
									onclick={handleImportClick}
									disabled={running || importing}
									class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
								>
									{importing ? 'importing...' : 'import results zip'}
								</button>
							</div>
							{#if datasetImportInfo}
								<div class="rounded border border-gray-800 bg-black/30 overflow-hidden">
									<button
										type="button"
										onclick={() => (datasetInspectorOpen = !datasetInspectorOpen)}
										class="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-gray-950/40"
									>
										<div class="space-y-1">
											<p class="text-[10px] font-mono text-gray-300">{datasetInspectorTitle()}</p>
											<p class="text-[10px] font-mono text-gray-500">{datasetInspectorSubtitle()}</p>
										</div>
										<span class="text-[10px] font-mono text-gray-500">{datasetInspectorOpen ? 'hide' : 'show'}</span>
									</button>
									{#if datasetInspectorOpen}
										<div class="border-t border-gray-800 p-3 space-y-3">
											<div class="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
												<div class="space-y-1">
													<p class="text-[10px] font-mono text-gray-400">
														schema: {datasetImportInfo.featureNames.join(', ') || 'no string fields detected'}
													</p>
													<p class="text-[10px] font-mono text-gray-500">
														splits: {datasetImportInfo.splitNames.join(', ')}
													</p>
												</div>
												<button
													type="button"
													onclick={requestDatasetRecipeProposal}
													disabled={datasetProposalBusy || !readyResearchProfile || running || importing || datasetBusy}
													class="rounded border border-blue-800 px-3 py-1.5 font-mono text-[10px] text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:opacity-40 transition-colors"
												>
													{datasetProposalBusy ? 'asking backend...' : 'suggest recipe'}
												</button>
											</div>
											{#if datasetProposalError}
												<p class="text-[10px] font-mono text-amber-300">{datasetProposalError}</p>
											{/if}
											<div class="grid gap-3 lg:grid-cols-2">
												<div class="rounded border border-gray-800 bg-gray-950/40 p-2 space-y-2">
													<p class="text-[10px] font-mono text-gray-400">
														train preview · {formatOptionalCount(datasetImportInfo.trainExamples)}
													</p>
													{#if datasetImportInfo.trainSamples.length === 0}
														<p class="text-[10px] font-mono text-gray-600">no train sample rows available</p>
													{:else}
														{#each datasetImportInfo.trainSamples as row}
															<pre class="overflow-x-auto rounded bg-black/40 p-2 text-[10px] font-mono text-gray-500">{JSON.stringify(row, null, 2)}</pre>
														{/each}
													{/if}
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 p-2 space-y-2">
													<p class="text-[10px] font-mono text-gray-400">
														val preview · {formatOptionalCount(datasetImportInfo.validationExamples)}
													</p>
													{#if datasetImportInfo.validationSamples.length === 0}
														<p class="text-[10px] font-mono text-gray-600">
															{datasetImportInfo.hasValidationSplit ? 'no validation sample rows available' : 'no validation split exposed by Hugging Face'}
														</p>
													{:else}
														{#each datasetImportInfo.validationSamples as row}
															<pre class="overflow-x-auto rounded bg-black/40 p-2 text-[10px] font-mono text-gray-500">{JSON.stringify(row, null, 2)}</pre>
														{/each}
													{/if}
												</div>
											</div>
											{#if datasetRecipeDraft}
												<div class="rounded border border-emerald-900/70 bg-emerald-950/10 p-3 space-y-3">
													<div class="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
														<div>
															<p class="text-[10px] font-mono text-emerald-300">recipe draft</p>
															<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.reasoning}</p>
														</div>
														<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.validationStrategy === 'huggingface' ? 'using hf validation split' : 'validation carved from train'}</p>
													</div>
													<div class="grid gap-2 md:grid-cols-2">
														<label class="space-y-1 md:col-span-2">
															<span class="block text-[10px] font-mono text-gray-500">label</span>
															<input
																type="text"
																value={datasetRecipeDraft.label}
																oninput={(event) => updateDatasetRecipeDraft({ label: (event.currentTarget as HTMLInputElement).value })}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">recipe</span>
															<select
																value={datasetRecipeDraft.recipeKey}
																onchange={(event) => applyRecipeOptionToDraft((event.currentTarget as HTMLSelectElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															>
																{#each datasetImportInfo.recipeOptions as option}
																	<option value={option.key}>{option.label}</option>
																{/each}
															</select>
															<p class="text-[10px] font-mono text-gray-600">{availableRecipeDescription(datasetRecipeDraft.recipeKey)}</p>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">text fields</span>
															<input
																type="text"
																value={datasetRecipeDraft.textFields.join(', ')}
																oninput={(event) => updateDatasetRecipeDraftTextFields((event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">train rows</span>
															<input
																type="text"
																inputmode="numeric"
																value={datasetRecipeDraft.maxTrainExamples == null ? '-' : String(datasetRecipeDraft.maxTrainExamples)}
																oninput={(event) => updateDatasetRecipeDraftCounts('maxTrainExamples', (event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">val rows</span>
															<input
																type="text"
																inputmode="numeric"
																value={datasetRecipeDraft.maxValidationExamples == null ? '-' : String(datasetRecipeDraft.maxValidationExamples)}
																oninput={(event) => updateDatasetRecipeDraftCounts('maxValidationExamples', (event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1 md:col-span-2">
															<span class="block text-[10px] font-mono text-gray-500">sample prompt</span>
															<textarea
																rows="4"
																value={datasetRecipeDraft.samplePrompt}
																oninput={(event) => updateDatasetRecipeDraft({ samplePrompt: (event.currentTarget as HTMLTextAreaElement).value })}
																disabled={running || importing || datasetBusy}
																class="w-full resize-y rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															></textarea>
														</label>
													</div>
													<div class="space-y-1">
														<p class="text-[10px] font-mono text-gray-400">preprocessing</p>
														<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.preprocessingSummary}</p>
														{#each datasetRecipeDraft.preprocessingSteps as step}
															<p class="text-[10px] font-mono text-gray-600">- {step}</p>
														{/each}
													</div>
												</div>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
							{#if datasetVersions.length > 0}
								<select
									value={activeDatasetVersionId ?? ''}
									onchange={(event) => activateDatasetVersion(Number((event.currentTarget as HTMLSelectElement).value))}
									disabled={running || importing || datasetBusy}
									class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
								>
									{#each datasetVersions as version}
										<option value={version.id}>
											{formatDatasetVersionSummary(version)}
										</option>
									{/each}
								</select>
							{/if}
						</div>
					{:else if setupTab === 'importResults'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">import results</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Restore experiments, metrics, inferences, and saved weights from a previous export ZIP.
								</p>
							</div>
							<div class="rounded border border-gray-800 bg-black/30 p-4 space-y-3">
								<div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
									<div class="space-y-1">
										<p class="text-[10px] font-mono text-gray-300">results archive</p>
										<p class="text-[10px] font-mono text-gray-500">
											Use an export produced by this app to continue work on another machine or browser.
										</p>
									</div>
									<button
										type="button"
										onclick={handleImportClick}
										disabled={running || importing}
										class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
									>
										{importing ? 'importing...' : 'choose results zip'}
									</button>
								</div>
								<p class="text-[10px] font-mono text-gray-500">
									This tab only restores experiment history and artifacts. It does not create a new raw dataset version.
								</p>
							</div>
						</div>
					{:else if setupTab === 'exportResults'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">export results</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Export one experiment workspace and its run history as a ZIP for backup, transfer, or continuation elsewhere.
								</p>
							</div>
							<div class="rounded border border-gray-800 bg-black/30 p-4 space-y-3">
								<label class="space-y-1">
									<span class="block text-[10px] font-mono text-gray-500">experiment workspace</span>
									<select
										bind:value={exportWorkspaceId}
										disabled={running || importing || workspaces.length === 0}
										class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
									>
										{#if workspaces.length === 0}
											<option value="">no experiment workspaces</option>
										{:else}
											{#each workspaces as workspace}
												<option value={workspace.id}>{formatWorkspaceSummary(workspace)}</option>
											{/each}
										{/if}
									</select>
								</label>
								<div class="flex items-center justify-between gap-3">
									<p class="text-[10px] font-mono text-gray-500">
										The ZIP contains experiment rows, loss history, inferences, and saved weights for the selected workspace.
									</p>
									<button
										type="button"
										onclick={handleExport}
										disabled={running || importing || !exportWorkspaceId}
										class="rounded border border-blue-800 px-3 py-1.5 font-mono text-xs text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:opacity-40 transition-colors"
									>
										export zip
									</button>
								</div>
							</div>
						</div>
					{:else}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">research backends</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									{backendSummary}. Configure a backend here before you ask the app to write or iterate on training code.
								</p>
							</div>
							<EndpointManager bind:profiles={researchProfiles} bind:selectedId={selectedResearchProfileId} disabled={running || importing} showHeader={false} />
						</div>
					{/if}
				</div>
			</div>
			<div class="rounded border border-gray-800 bg-gray-950/60 p-6 flex flex-col items-center justify-center space-y-4 mx-auto" style="min-height: calc(100vh - 18rem);">
				<input bind:this={importInput} type="file" accept=".zip,application/zip" class="hidden" onchange={handleImportFile} />
				{#if status === 'initializing' || status === 'loading dataset...'}
					<p class="text-sm font-mono text-gray-500">{status}</p>
				{:else}
					<div class="flex flex-col items-center gap-3">
						<button
							onclick={() => { mode = 'research'; startResearch(); }}
							disabled={!canTrain || running || importing || datasetBusy}
							class="rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-8 py-4 font-mono text-sm text-white transition-colors"
						>
							start research
						</button>
						<button
							onclick={handleImportClick}
							disabled={running || importing}
							class="rounded border border-gray-700 px-5 py-2 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
						>
							{importing ? 'importing...' : 'import results zip'}
						</button>
					</div>
					<p class="text-xs font-mono text-gray-500 max-w-sm text-center">
						Configure a backend to run new research, or import a prior export ZIP to restore experiment history from another machine.
					</p>
					{#if status !== 'ready'}
						<p class="text-sm font-mono text-amber-300 text-center">{status}</p>
					{/if}
				{/if}
			</div>
		</div>
		{:else}
		<div class="space-y-4">
			<div class="rounded border border-gray-800 bg-gray-950/60 overflow-hidden">
				<div class="border-b border-gray-800 px-3 pt-3">
					<div class="flex items-center gap-2">
						<button
							type="button"
							onclick={() => (setupTab = 'experiments')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'experiments' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>experiments</button>
						<button
							type="button"
							onclick={() => (setupTab = 'createExperiment')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'createExperiment' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>create experiment</button>
						<button
							type="button"
							onclick={() => (setupTab = 'importDataset')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'importDataset' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>import dataset</button>
						<button
							type="button"
							onclick={() => (setupTab = 'importResults')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'importResults' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>import results</button>
						<button
							type="button"
							onclick={() => (setupTab = 'exportResults')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'exportResults' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>export results</button>
						<button
							type="button"
							onclick={() => (setupTab = 'research')}
							class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {setupTab === 'research' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
						>research backends</button>
					</div>
				</div>
				<div class="p-3">
					{#if setupTab === 'experiments'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">experiments</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Choose the active experiment workspace. The chart, leaderboard, inference panel, and resume actions below all follow this selection.
								</p>
							</div>
							{#if workspaces.length === 0}
								<div class="rounded border border-dashed border-gray-700 bg-black/20 p-4 space-y-3">
									<p class="text-xs font-mono text-gray-300">no experiment workspaces created</p>
									<p class="text-[10px] font-mono text-gray-500">
										Import a dataset or results bundle first, then create an experiment that assigns one of those reusable bases.
									</p>
									<div class="flex flex-wrap gap-2">
										<button
											type="button"
											onclick={() => (setupTab = 'createExperiment')}
											class="rounded border border-blue-800 px-3 py-1.5 font-mono text-xs text-blue-300 hover:border-blue-600 hover:text-blue-200 transition-colors"
										>
											create experiment
										</button>
										<button
											type="button"
											onclick={() => (setupTab = 'importDataset')}
											class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white transition-colors"
										>
											import dataset
										</button>
										<button
											type="button"
											onclick={() => (setupTab = 'importResults')}
											class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white transition-colors"
										>
											import results
										</button>
									</div>
								</div>
							{:else}
								<div class="grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
									<div class="rounded border border-gray-800 bg-black/30 p-3 space-y-3">
										<div class="flex flex-col gap-2 md:flex-row md:items-end">
											<label class="min-w-0 flex-1 space-y-1">
												<span class="block text-[10px] font-mono text-gray-500">active experiment</span>
												<select
													value={currentWorkspaceId ?? ''}
													onchange={(event) => selectWorkspace(Number((event.currentTarget as HTMLSelectElement).value))}
													disabled={running || importing || datasetBusy}
													class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
												>
													{#each workspaces as workspace}
														<option value={workspace.id}>{formatWorkspaceSummary(workspace)}</option>
													{/each}
												</select>
											</label>
											<button
												type="button"
												onclick={() => (setupTab = 'createExperiment')}
												class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white transition-colors"
											>
												new experiment
											</button>
										</div>
										{#if currentWorkspace}
											<div class="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
												<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
													<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">source</p>
													<p class="mt-1 text-[11px] font-mono text-gray-300">{currentWorkspace.source_type}</p>
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
													<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">assigned dataset</p>
													<p class="mt-1 text-[11px] font-mono text-gray-300">{activeDataset ? formatDatasetVersionSummary(activeDataset) : 'none'}</p>
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
													<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">runs</p>
													<p class="mt-1 text-[11px] font-mono text-gray-300">{experiments.length}</p>
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
													<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">created</p>
													<p class="mt-1 text-[11px] font-mono text-gray-300">{new Date(currentWorkspace.created_at).toLocaleString()}</p>
												</div>
											</div>
											{#if currentWorkspaceBaseExperiment}
												<div class="rounded border border-gray-800 bg-gray-950/40 px-3 py-2">
													<p class="text-[10px] font-mono text-gray-400">results baseline</p>
													<p class="mt-1 text-[11px] font-mono text-gray-300">{formatExperimentOption(currentWorkspaceBaseExperiment)}</p>
												</div>
											{/if}
										{/if}
									</div>
									<div class="rounded border border-gray-800 bg-black/30 p-3 space-y-3">
										{#if currentWorkspace}
											<div>
												<p class="text-[10px] font-mono text-gray-400">readme</p>
												<p class="mt-1 whitespace-pre-wrap text-[11px] font-mono text-gray-300">{currentWorkspace.readme.trim() || 'no readme yet'}</p>
											</div>
											<div>
												<p class="text-[10px] font-mono text-gray-400">notes</p>
												<p class="mt-1 whitespace-pre-wrap text-[11px] font-mono text-gray-500">{currentWorkspace.notes.trim() || 'no private notes yet'}</p>
											</div>
										{/if}
									</div>
								</div>
								<div class="rounded border border-gray-800 bg-black/30 overflow-hidden">
									<button
										type="button"
										onclick={() => (activeDatasetReviewOpen = !activeDatasetReviewOpen)}
										class="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-gray-950/40"
									>
										<div class="space-y-1">
											<p class="text-[10px] font-mono text-gray-300">
												dataset + recipe
												{#if activeDataset}
													· #{datasetVersionNumberById.get(activeDataset.id) ?? activeDataset.id} · {activeDataset.label}
												{/if}
											</p>
											<p class="text-[10px] font-mono text-gray-500">
												{#if activeDataset}
													{activeDataset.trainExamples.toLocaleString()} train · {activeDataset.validationExamples.toLocaleString()} val · recipe {activeDataset.recipeKey}
												{:else}
													no dataset assigned to the selected experiment
												{/if}
											</p>
										</div>
										<div class="flex items-center gap-3">
											<span class="text-[10px] font-mono {activeDatasetPrepReady ? 'text-emerald-300' : 'text-amber-300'}">
												{activeDatasetPrepReady ? 'reviewed for start' : 'review required before next start'}
											</span>
											<span class="text-[10px] font-mono text-gray-500">{activeDatasetReviewOpen ? 'hide' : 'show'}</span>
										</div>
									</button>
									{#if activeDatasetReviewOpen}
										<div class="border-t border-gray-800 p-3 space-y-3">
											{#if activeDataset}
												<div class="space-y-1 text-[10px] font-mono text-gray-500">
													<p title={`db id ${activeDataset.id}`}>{activeDataset.sourceRef} · trainer {activeDataset.trainerLabel} · family {activeDataset.modelFamily}</p>
													<p>{activeDataset.sourceType === 'legacy' ? 'legacy static binaries' : `config ${activeDataset.configName}${activeDataset.revision ? ` · ${activeDataset.revision}` : ''}`}</p>
												</div>
												<div class="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">recipe</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{activeDataset.recipeDescription}</p>
													</div>
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">text fields</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{activeDataset.textFields.join(', ') || 'none'}</p>
													</div>
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">splits</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{activeDataset.trainExamples.toLocaleString()} train · {activeDataset.validationExamples.toLocaleString()} val</p>
													</div>
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">row caps</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{formatOptionalCount(activeDataset.maxTrainExamples)} train · {formatOptionalCount(activeDataset.maxValidationExamples)} val</p>
													</div>
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">bytes</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{Math.round(activeDataset.trainBytes / 1024)} KB train · {Math.round(activeDataset.validationBytes / 1024)} KB val</p>
													</div>
													<div class="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
														<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">prompt preset</p>
														<p class="mt-1 text-[11px] font-mono text-gray-300">{activeDataset.samplePrompt.trim() || 'none'}</p>
													</div>
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 p-3 space-y-2">
													<p class="text-[10px] font-mono text-gray-400">preprocessing</p>
													<p class="text-[10px] font-mono text-gray-500">{activeDataset.preprocessingSummary}</p>
													{#each activeDataset.preprocessingSteps as step}
														<p class="text-[10px] font-mono text-gray-600">- {step}</p>
													{/each}
												</div>
											{:else}
												<p class="text-[10px] font-mono text-amber-300">
													This experiment does not currently resolve to a dataset. Create it from a prepared dataset, or use a results baseline that still carries dataset metadata.
												</p>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{:else if setupTab === 'createExperiment'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">create experiment</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Create a reusable experiment workspace from a prepared dataset or an imported results baseline.
								</p>
							</div>
							<div class="rounded border border-gray-800 bg-black/30 p-3 space-y-3">
								<div class="flex flex-wrap gap-2">
									<button
										type="button"
										onclick={() => (createExperimentSourceType = 'dataset')}
										class="rounded border px-3 py-1.5 font-mono text-xs transition-colors {createExperimentSourceType === 'dataset' ? 'border-blue-700 bg-blue-950/30 text-blue-200' : 'border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white'}"
									>
										prepared dataset
									</button>
									<button
										type="button"
										onclick={() => (createExperimentSourceType = 'results')}
										class="rounded border px-3 py-1.5 font-mono text-xs transition-colors {createExperimentSourceType === 'results' ? 'border-blue-700 bg-blue-950/30 text-blue-200' : 'border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white'}"
									>
										imported results
									</button>
								</div>
								{#if createExperimentSourceType === 'dataset'}
									<label class="space-y-1">
										<span class="block text-[10px] font-mono text-gray-500">dataset + recipe base</span>
										<select
											bind:value={createExperimentDatasetId}
											disabled={running || importing || datasetBusy || datasetVersions.length === 0}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
										>
											<option value="">use active dataset</option>
											{#each datasetVersions as version}
												<option value={version.id}>{formatDatasetVersionSummary(version)}</option>
											{/each}
										</select>
									</label>
								{:else}
									<label class="space-y-1">
										<span class="block text-[10px] font-mono text-gray-500">results baseline</span>
										<select
											bind:value={createExperimentBaseExperimentId}
											disabled={running || importing || allExperimentOptions.length === 0}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
										>
											<option value="">select imported experiment</option>
											{#each allExperimentOptions as exp}
												<option value={exp.id}>{formatExperimentOption(exp)}</option>
											{/each}
										</select>
									</label>
								{/if}
								<div class="grid gap-2 md:grid-cols-2">
									<label class="space-y-1 md:col-span-2">
										<span class="block text-[10px] font-mono text-gray-500">name</span>
										<input
											type="text"
											bind:value={createExperimentName}
											placeholder="irishman control study"
											disabled={createExperimentBusy}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
										/>
									</label>
									<label class="space-y-1 md:col-span-2">
										<span class="block text-[10px] font-mono text-gray-500">readme</span>
										<textarea
											rows="4"
											bind:value={createExperimentReadme}
											placeholder="Goal, benchmark prompts, success criteria."
											disabled={createExperimentBusy}
											class="w-full resize-y rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
										></textarea>
									</label>
									<label class="space-y-1 md:col-span-2">
										<span class="block text-[10px] font-mono text-gray-500">private notes</span>
										<textarea
											rows="3"
											bind:value={createExperimentNotes}
											placeholder="Anything you want to remember while iterating."
											disabled={createExperimentBusy}
											class="w-full resize-y rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
										></textarea>
									</label>
								</div>
								<div class="flex items-center justify-between gap-3">
									<p class="text-[10px] font-mono text-gray-500">
										{createExperimentSourceType === 'dataset'
											? 'Use this when you want several independent experiment branches on the same prepared dataset + recipe.'
											: 'Use this when you want to continue from imported experiment history and reuse its baseline code context.'}
									</p>
									<button
										type="button"
										onclick={handleCreateWorkspace}
										disabled={createExperimentBusy}
										class="rounded border border-blue-800 px-3 py-1.5 font-mono text-xs text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:opacity-40 transition-colors"
									>
										{createExperimentBusy ? 'creating...' : 'create experiment'}
									</button>
								</div>
							</div>
						</div>
					{:else if setupTab === 'importDataset'}
						<div class="space-y-3">
							<div class="flex flex-col gap-2 xl:flex-row xl:items-start xl:justify-between">
								<div>
									<h2 class="text-xs font-mono text-gray-300">import dataset</h2>
									<p class="mt-1 text-[10px] font-mono text-gray-500">
										Inspect a Hugging Face dataset, edit the proposed recipe, then materialize a new local dataset version.
									</p>
								</div>
								<div class="grid w-full grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_110px_110px_auto] md:items-end">
									<label class="block">
										<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">dataset id</span>
										<input
											type="text"
											bind:value={datasetImportId}
											placeholder="huggingface dataset id"
											disabled={running || importing || datasetBusy}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
										/>
									</label>
									<label class="block">
										<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">{datasetTrainLabel()}</span>
										<input
											type="text"
											inputmode="numeric"
											bind:value={datasetImportMaxTrain}
											disabled={running || importing || datasetBusy}
											placeholder={datasetImportSuggestedTrain || 'train rows'}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
											title="max train examples"
										/>
									</label>
									<label class="block">
										<span class="mb-1 block text-[10px] font-mono uppercase tracking-wide text-gray-500">{datasetValLabel()}</span>
										<input
											type="text"
											inputmode="numeric"
											bind:value={datasetImportMaxValidation}
											disabled={running || importing || datasetBusy}
											placeholder={datasetImportSuggestedValidation || '-'}
											class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
											title="max validation examples"
										/>
									</label>
									<button
										type="button"
										onclick={importHuggingFaceDataset}
										disabled={running || importing || datasetBusy}
										class="rounded border border-emerald-800 px-3 py-1.5 font-mono text-xs text-emerald-300 hover:border-emerald-600 hover:text-emerald-200 disabled:opacity-40 transition-colors"
									>
										{datasetBusy ? 'syncing...' : 'import hf'}
									</button>
								</div>
							</div>
							<p class="text-[10px] font-mono {datasetImportInfoError ? 'text-amber-300' : 'text-gray-500'}">
								{datasetImportHelperText()}
							</p>
							{#if datasetImportInfo}
								<div class="rounded border border-gray-800 bg-black/30 overflow-hidden">
									<button
										type="button"
										onclick={() => (datasetInspectorOpen = !datasetInspectorOpen)}
										class="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-gray-950/40"
									>
										<div class="space-y-1">
											<p class="text-[10px] font-mono text-gray-300">{datasetInspectorTitle()}</p>
											<p class="text-[10px] font-mono text-gray-500">{datasetInspectorSubtitle()}</p>
										</div>
										<span class="text-[10px] font-mono text-gray-500">{datasetInspectorOpen ? 'hide' : 'show'}</span>
									</button>
									{#if datasetInspectorOpen}
										<div class="border-t border-gray-800 p-3 space-y-3">
											<div class="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
												<div class="space-y-1">
													<p class="text-[10px] font-mono text-gray-400">
														schema: {datasetImportInfo.featureNames.join(', ') || 'no string fields detected'}
													</p>
													<p class="text-[10px] font-mono text-gray-500">
														splits: {datasetImportInfo.splitNames.join(', ')}
													</p>
												</div>
												<button
													type="button"
													onclick={requestDatasetRecipeProposal}
													disabled={datasetProposalBusy || !readyResearchProfile || running || importing || datasetBusy}
													class="rounded border border-blue-800 px-3 py-1.5 font-mono text-[10px] text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:opacity-40 transition-colors"
												>
													{datasetProposalBusy ? 'asking backend...' : 'suggest recipe'}
												</button>
											</div>
											{#if datasetProposalError}
												<p class="text-[10px] font-mono text-amber-300">{datasetProposalError}</p>
											{/if}
											<div class="grid gap-3 xl:grid-cols-2">
												<div class="rounded border border-gray-800 bg-gray-950/40 p-2 space-y-2">
													<p class="text-[10px] font-mono text-gray-400">
														train preview · {formatOptionalCount(datasetImportInfo.trainExamples)}
													</p>
													{#if datasetImportInfo.trainSamples.length === 0}
														<p class="text-[10px] font-mono text-gray-600">no train sample rows available</p>
													{:else}
														{#each datasetImportInfo.trainSamples as row}
															<pre class="overflow-x-auto rounded bg-black/40 p-2 text-[10px] font-mono text-gray-500">{JSON.stringify(row, null, 2)}</pre>
														{/each}
													{/if}
												</div>
												<div class="rounded border border-gray-800 bg-gray-950/40 p-2 space-y-2">
													<p class="text-[10px] font-mono text-gray-400">
														val preview · {formatOptionalCount(datasetImportInfo.validationExamples)}
													</p>
													{#if datasetImportInfo.validationSamples.length === 0}
														<p class="text-[10px] font-mono text-gray-600">
															{datasetImportInfo.hasValidationSplit ? 'no validation sample rows available' : 'no validation split exposed by Hugging Face'}
														</p>
													{:else}
														{#each datasetImportInfo.validationSamples as row}
															<pre class="overflow-x-auto rounded bg-black/40 p-2 text-[10px] font-mono text-gray-500">{JSON.stringify(row, null, 2)}</pre>
														{/each}
													{/if}
												</div>
											</div>
											{#if datasetRecipeDraft}
												<div class="rounded border border-emerald-900/70 bg-emerald-950/10 p-3 space-y-3">
													<div class="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
														<div>
															<p class="text-[10px] font-mono text-emerald-300">recipe draft</p>
															<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.reasoning}</p>
														</div>
														<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.validationStrategy === 'huggingface' ? 'using hf validation split' : 'validation carved from train'}</p>
													</div>
													<div class="grid gap-2 md:grid-cols-2">
														<label class="space-y-1 md:col-span-2">
															<span class="block text-[10px] font-mono text-gray-500">label</span>
															<input
																type="text"
																value={datasetRecipeDraft.label}
																oninput={(event) => updateDatasetRecipeDraft({ label: (event.currentTarget as HTMLInputElement).value })}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">recipe</span>
															<select
																value={datasetRecipeDraft.recipeKey}
																onchange={(event) => applyRecipeOptionToDraft((event.currentTarget as HTMLSelectElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															>
																{#each datasetImportInfo.recipeOptions as option}
																	<option value={option.key}>{option.label}</option>
																{/each}
															</select>
															<p class="text-[10px] font-mono text-gray-600">{availableRecipeDescription(datasetRecipeDraft.recipeKey)}</p>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">text fields</span>
															<input
																type="text"
																value={datasetRecipeDraft.textFields.join(', ')}
																oninput={(event) => updateDatasetRecipeDraftTextFields((event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">train rows</span>
															<input
																type="text"
																inputmode="numeric"
																value={datasetRecipeDraft.maxTrainExamples == null ? '-' : String(datasetRecipeDraft.maxTrainExamples)}
																oninput={(event) => updateDatasetRecipeDraftCounts('maxTrainExamples', (event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1">
															<span class="block text-[10px] font-mono text-gray-500">val rows</span>
															<input
																type="text"
																inputmode="numeric"
																value={datasetRecipeDraft.maxValidationExamples == null ? '-' : String(datasetRecipeDraft.maxValidationExamples)}
																oninput={(event) => updateDatasetRecipeDraftCounts('maxValidationExamples', (event.currentTarget as HTMLInputElement).value)}
																disabled={running || importing || datasetBusy}
																class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															/>
														</label>
														<label class="space-y-1 md:col-span-2">
															<span class="block text-[10px] font-mono text-gray-500">sample prompt</span>
															<textarea
																rows="4"
																value={datasetRecipeDraft.samplePrompt}
																oninput={(event) => updateDatasetRecipeDraft({ samplePrompt: (event.currentTarget as HTMLTextAreaElement).value })}
																disabled={running || importing || datasetBusy}
																class="w-full resize-y rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
															></textarea>
														</label>
													</div>
													<div class="space-y-1">
														<p class="text-[10px] font-mono text-gray-400">preprocessing</p>
														<p class="text-[10px] font-mono text-gray-500">{datasetRecipeDraft.preprocessingSummary}</p>
														{#each datasetRecipeDraft.preprocessingSteps as step}
															<p class="text-[10px] font-mono text-gray-600">- {step}</p>
														{/each}
													</div>
												</div>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{:else if setupTab === 'importResults'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">import results</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Restore experiments, metrics, inferences, and saved weights from a previous export ZIP.
								</p>
							</div>
							<div class="rounded border border-gray-800 bg-black/30 p-4 space-y-3">
								<div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
									<div class="space-y-1">
										<p class="text-[10px] font-mono text-gray-300">results archive</p>
										<p class="text-[10px] font-mono text-gray-500">
											Use an export produced by this app to continue work on another machine or browser.
										</p>
									</div>
									<button
										type="button"
										onclick={handleImportClick}
										disabled={running || importing}
										class="rounded border border-gray-700 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
									>
										{importing ? 'importing...' : 'choose results zip'}
									</button>
								</div>
								<p class="text-[10px] font-mono text-gray-500">
									This tab only restores experiment history and artifacts. It does not create a new raw dataset version.
								</p>
							</div>
						</div>
					{:else if setupTab === 'exportResults'}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">export results</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									Export one experiment workspace and its run history as a ZIP for backup, transfer, or continuation elsewhere.
								</p>
							</div>
							<div class="rounded border border-gray-800 bg-black/30 p-4 space-y-3">
								<label class="space-y-1">
									<span class="block text-[10px] font-mono text-gray-500">experiment workspace</span>
									<select
										bind:value={exportWorkspaceId}
										disabled={running || importing || workspaces.length === 0}
										class="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 font-mono text-xs text-gray-200 disabled:opacity-40"
									>
										{#if workspaces.length === 0}
											<option value="">no experiment workspaces</option>
										{:else}
											{#each workspaces as workspace}
												<option value={workspace.id}>{formatWorkspaceSummary(workspace)}</option>
											{/each}
										{/if}
									</select>
								</label>
								<div class="flex items-center justify-between gap-3">
									<p class="text-[10px] font-mono text-gray-500">
										The ZIP contains workspace metadata, experiment rows, loss history, inferences, and saved weights for the selected workspace.
									</p>
									<button
										type="button"
										onclick={handleExport}
										disabled={running || importing || !exportWorkspaceId}
										class="rounded border border-blue-800 px-3 py-1.5 font-mono text-xs text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:opacity-40 transition-colors"
									>
										export zip
									</button>
								</div>
							</div>
						</div>
					{:else}
						<div class="space-y-3">
							<div>
								<h2 class="text-xs font-mono text-gray-300">research backends</h2>
								<p class="mt-1 text-[10px] font-mono text-gray-500">
									{backendSummary}. Pick the backend the app should use when it writes or critiques training code.
								</p>
							</div>
							<EndpointManager bind:profiles={researchProfiles} bind:selectedId={selectedResearchProfileId} disabled={running || importing} showHeader={false} />
						</div>
					{/if}
				</div>
			</div>

			<div class="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(24rem,0.95fr)] 2xl:grid-cols-[minmax(0,1.55fr)_minmax(28rem,1fr)]">
				<div class="flex flex-col gap-4 min-w-0">
					<div class="rounded border border-gray-800 bg-gray-950/60 p-3 space-y-3">
						<div class="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
							<div class="flex items-center gap-3">
								<div class="flex rounded border border-gray-700 text-xs font-mono overflow-hidden">
									<button
										class="px-2.5 py-0.5 {mode === 'manual' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}"
										onclick={() => (mode = 'manual')}
										disabled={running}
									>manual</button>
									<button
										class="px-2.5 py-0.5 {mode === 'research' ? 'bg-blue-700 text-white' : 'text-gray-400 hover:text-white'}"
										onclick={() => (mode = 'research')}
										disabled={running}
									>auto</button>
								</div>
								<p class="text-[11px] font-mono text-gray-500">{displayStatus}</p>
							</div>
							<div class="flex flex-col gap-2 md:flex-row md:items-center md:justify-end md:flex-1">
								{#if mode === 'manual'}
									<input
										type="text"
										bind:value={experimentName}
										placeholder="experiment name..."
										disabled={running}
										class="w-full md:w-56 bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
									/>
								{/if}
								<button
									onclick={() => (code = defaultBaselineCode)}
									disabled={running}
									class="rounded border border-gray-700 px-3 py-1.5 text-[10px] font-mono text-gray-400 hover:text-white hover:border-gray-500 disabled:opacity-30"
								>reset code</button>
								{#if running}
									<button
										onclick={stopCurrentRun}
										disabled={stopIntent === 'immediate'}
										class="rounded px-3 py-1.5 font-mono text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-60 {stopIntent === 'graceful' ? 'bg-amber-600 hover:bg-amber-500 text-black' : 'bg-red-600 hover:bg-red-500 text-white'}"
									>
										{stopButtonLabel}
									</button>
								{:else if mode === 'manual'}
									<button onclick={startManualTraining}
										disabled={!canTrain || importing || datasetBusy}
										class="rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors">
										{activeDatasetPrepReady ? 'run train.ts' : 'review prep + run'}
									</button>
								{:else}
									<button onclick={startResearch}
										disabled={!canTrain || importing || datasetBusy}
										class="rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors">
										{activeDatasetPrepReady ? 'start research' : 'review prep + start'}
									</button>
								{/if}
							</div>
						</div>

						<div class="rounded border border-gray-800 bg-black/20 p-3 min-h-[6.5rem]">
							{#if selectedExp}
								<div class="flex items-center gap-2 font-mono text-xs">
									<span class="px-1 py-0.5 rounded text-[10px] {selectedExp.source === 'auto' ? 'bg-blue-900/50 text-blue-300' : 'bg-gray-700 text-gray-300'}">
										{selectedExp.source === 'auto' ? 'auto' : 'manual'}
									</span>
									{#if selectedRunNumber != null}
										<span class="text-[10px] text-gray-500 tabular-nums" title={`dataset-local run ${selectedRunNumber} · db id ${selectedExp.id}`}>
											run {selectedRunNumber}
										</span>
									{/if}
									<span class="text-gray-200">{selectedExp.name}</span>
									<div class="ml-auto flex items-center gap-3 text-[10px] font-mono text-gray-500">
										{#if selectedPrimaryMetric?.value != null}
											<span class="tabular-nums">{selectedPrimaryMetric.label}: {formatExperimentMetricValue(selectedExp)}</span>
										{/if}
										<span class="tabular-nums">{selectedExp.valBpb.toFixed(4)} {activeTrainer.metricShortLabel}</span>
									</div>
								</div>
								{#if selectedExp.rerunOf || selectedExp.benchmarkGroup}
									<p class="mt-2 text-[10px] text-amber-300 font-mono">
										{#if selectedExp.rerunOf}rerun of #{selectedExp.rerunOf}{/if}
										{#if selectedExp.rerunOf && selectedExp.benchmarkGroup} · {/if}
										{#if selectedExp.benchmarkGroup}baseline {selectedExp.benchmarkGroup}{/if}
									</p>
								{/if}
								{#if selectedExp.reasoning}
									<p class="mt-2 text-[11px] text-gray-400 font-mono line-clamp-2" title={selectedExp.reasoning}>{selectedExp.reasoning}</p>
								{/if}
								{#if selectedExp.error}
									<p class="mt-2 text-[11px] text-red-400 font-mono line-clamp-2">error: {selectedExp.error}</p>
								{/if}
								{#if selectedEvalMetricRows.length > 0}
									<div class="mt-3 grid grid-cols-2 gap-2 xl:grid-cols-3">
										{#each selectedEvalMetricRows as row}
											<div class="rounded border border-gray-800 bg-black/20 px-2 py-1.5">
												<p class="text-[9px] font-mono uppercase tracking-[0.12em] text-gray-600">{row.key}</p>
												<p class="mt-1 text-[11px] font-mono text-gray-300" title={row.label}>{row.value}</p>
											</div>
										{/each}
									</div>
								{/if}
								<div class="mt-3 flex items-center gap-2">
									<button
										onclick={rerunSelectedExperiment}
										disabled={running || importing}
										class="rounded border border-gray-700 px-2 py-1 font-mono text-[10px] text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
									>
										rerun this code
									</button>
									<button
										onclick={deleteSelectedExperiment}
										disabled={running || importing}
										class="rounded border border-red-900 px-2 py-1 font-mono text-[10px] text-red-300 hover:border-red-700 hover:text-red-200 disabled:opacity-40 transition-colors"
									>
										delete this run
									</button>
								</div>
							{:else if running}
								<div class="flex items-center gap-2 font-mono text-xs">
									<span class="px-1 py-0.5 rounded text-[10px] bg-blue-900/50 text-blue-300 animate-pulse">
										{waitingForRecommendation ? 'thinking' : 'running'}
									</span>
									<span class="text-gray-200">
										{waitingForRecommendation ? `${activeResearchLabel} writing code...` : currentRunName || 'training...'}
									</span>
								</div>
								{#if currentReasoning && !waitingForRecommendation}
									<p class="mt-2 text-[11px] text-gray-400 font-mono line-clamp-2" title={currentReasoning}>{currentReasoning}</p>
								{/if}
								<p class="mt-2 text-[11px] text-gray-500 font-mono">{displayStatus}</p>
							{:else}
								<div class="space-y-1">
									<h2 class="text-xs font-mono text-gray-400">workspace</h2>
									<p class="text-[11px] font-mono text-gray-500">
										Use the tabs below to inspect the loss chart, edit the training code, or run inference on the selected experiment.
									</p>
								</div>
							{/if}
						</div>
					</div>

					<div class="rounded border border-gray-800 bg-gray-950/60 flex flex-col min-h-[28rem] xl:h-[min(38rem,calc(100vh-22rem))] 2xl:h-[min(42rem,calc(100vh-20rem))] overflow-hidden">
						<div class="border-b border-gray-800 px-3 pt-3">
							<div class="flex items-center gap-2">
								<button
									type="button"
									onclick={() => (workspaceTab = 'chart')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'chart' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>chart</button>
								<button
									type="button"
									onclick={() => (workspaceTab = 'code')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'code' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>code</button>
								<button
									type="button"
									onclick={() => (workspaceTab = 'inference')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'inference' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>inference</button>
							</div>
						</div>

						<div class="flex-1 min-h-0 p-3">
							{#if workspaceTab === 'chart'}
								<div class="h-full flex flex-col gap-3">
									<div class="flex flex-col gap-2 xl:flex-row xl:items-start xl:justify-between">
										<div>
											<h2 class="text-xs font-mono text-gray-400">loss</h2>
											{#if selectedExp}
												<span class="text-[10px] font-mono text-gray-500 tabular-nums">
													{selectedExp.totalSteps} steps · {(selectedExp.elapsed / 1000).toFixed(1)}s
												</span>
											{/if}
										</div>
										<div class="flex flex-wrap items-center gap-2 text-[10px] font-mono text-gray-400">
											<select
												bind:value={chartSeriesMode}
												disabled={!canFocusChart}
												class="rounded border border-gray-800 bg-black/40 px-2 py-1 disabled:opacity-40"
												title="chart series view"
											>
												<option value="all">view: all runs</option>
												<option value="focus">view: focus selected</option>
											</select>
											<select
												bind:value={chartScaleMode}
												class="rounded border border-gray-800 bg-black/40 px-2 py-1"
												title="chart y-axis scaling"
											>
												<option value="fit">y: fit all</option>
												<option value="trim">y: trim outliers</option>
												<option value="manual">y: manual range</option>
											</select>
											{#if chartScaleMode === 'manual'}
												<input
													type="number"
													step="0.01"
													bind:value={chartYMinInput}
													placeholder="y min"
													class="w-20 rounded border border-gray-800 bg-black/40 px-2 py-1 text-right tabular-nums"
													title="manual minimum y-axis value"
												/>
												<input
													type="number"
													step="0.01"
													bind:value={chartYMaxInput}
													placeholder="y max"
													class="w-20 rounded border border-gray-800 bg-black/40 px-2 py-1 text-right tabular-nums"
													title="manual maximum y-axis value"
												/>
												{#if !chartManualRangeValid}
													<span class="text-amber-300">enter y-max greater than y-min</span>
												{/if}
											{/if}
										</div>
									</div>
									<div class="flex-1 min-h-[22rem] rounded border border-gray-800 bg-black/20 p-3">
										<LossChart
											data={lossData}
											pastRuns={pastLossRuns}
											seriesMode={chartSeriesMode}
											yScaleMode={chartScaleMode}
											yMin={chartYMin}
											yMax={chartYMax}
										/>
									</div>
								</div>
							{:else if workspaceTab === 'code'}
								<div class="h-full flex flex-col min-h-0">
									<div class="flex items-center justify-between mb-2 shrink-0">
										<h2 class="text-xs font-mono text-gray-400">train.ts</h2>
										<span class="text-[10px] font-mono text-gray-500">
											{running && mode === 'research' ? 'locked during auto runs' : 'editable'}
										</span>
									</div>
									<div class="flex-1 min-h-0">
										<CodeEditor bind:value={code} disabled={running && mode === 'research'} />
									</div>
								</div>
							{:else}
								<div class="h-full flex flex-col min-h-0 gap-2">
									<div class="flex items-center justify-between gap-3 shrink-0">
										<h2 class="text-xs font-mono text-gray-400">inference</h2>
										{#if selectedExp}
											<span class="text-[10px] font-mono text-gray-500" title={`db id ${selectedExp.id}`}>
												selected run {selectedRunNumber ?? '?'}
											</span>
										{/if}
									</div>
									{#if hasAnyModel}
										<div class="flex flex-col gap-2 shrink-0 md:flex-row md:items-start">
											<textarea
												bind:value={prompt}
												rows="4"
												placeholder={activeDatasetSamplePrompt || 'prompt...'}
												disabled={running || !selectedExpId || sampling}
												class="min-h-24 flex-1 resize-y bg-gray-800 border border-gray-700 rounded px-2 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
											></textarea>
											<div class="flex items-center gap-2 md:flex-col md:items-stretch">
												<button
													type="button"
													onclick={applyInferencePromptPreset}
													disabled={running || !selectedExpId || !activeDatasetSamplePrompt || sampling}
													class="rounded border border-blue-800 px-3 py-1 font-mono text-xs text-blue-300 hover:border-blue-600 hover:text-blue-200 disabled:border-gray-800 disabled:text-gray-500 transition-colors"
												>
													use preset
												</button>
												<input type="number" bind:value={temperature} min={0.1} max={2} step={0.1}
													disabled={running || !selectedExpId || sampling}
													class="w-16 bg-gray-800 border border-gray-700 rounded px-1 py-1 text-right tabular-nums text-xs text-gray-200 font-mono disabled:opacity-40"
													title="temperature" />
												<button onclick={generateSample} disabled={running || !selectedExpId || sampling}
													class="rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 px-3 py-1 font-mono text-xs transition-colors">
												{#if sampling}
													<svg class="animate-spin h-3.5 w-3.5 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
														<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
														<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
													</svg>
												{:else}
													go
												{/if}
												</button>
											</div>
										</div>
										{#if running}
											<p class="text-[10px] font-mono text-amber-300">
												inference is paused while training is running to avoid shared-state conflicts.
											</p>
										{/if}
										<div class="flex-1 min-h-0 overflow-y-auto rounded border border-gray-800 bg-black/20 p-3">
											{#if sampling && streamingOutput}
												<pre class="text-xs text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed">{streamingOutput}<span class="animate-pulse">▌</span></pre>
											{:else if currentInference}
												{#if inferences.length > 1}
													<div class="flex items-center justify-end text-xs font-mono text-gray-500 mb-1">
														<div class="flex items-center gap-1">
															<button onclick={() => { inferenceIdx = Math.min(inferenceIdx + 1, inferences.length - 1); }}
																disabled={inferenceIdx >= inferences.length - 1} class="px-1 hover:text-gray-300 disabled:opacity-30">←</button>
															<span>{inferences.length - inferenceIdx}/{inferences.length}</span>
															<button onclick={() => { inferenceIdx = Math.max(inferenceIdx - 1, 0); }}
																disabled={inferenceIdx <= 0} class="px-1 hover:text-gray-300 disabled:opacity-30">→</button>
														</div>
													</div>
												{/if}
												<pre class="text-xs text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed">{currentInference.output}</pre>
											{:else}
												<div class="h-full flex items-center justify-center text-center">
													<p class="text-[11px] font-mono text-gray-500 max-w-sm">
														Select an experiment with saved weights, then sample from it here.
													</p>
												</div>
											{/if}
										</div>
									{:else}
										<div class="flex-1 min-h-0 rounded border border-gray-800 bg-black/20 flex items-center justify-center text-center p-6">
											<p class="text-[11px] font-mono text-gray-500 max-w-sm">
												Train or import experiments first, then select one to use the inference workspace.
											</p>
										</div>
									{/if}
								</div>
							{/if}
						</div>
					</div>
				</div>

				<!-- Right: leaderboard -->
				<div class="flex flex-col xl:h-[min(38rem,calc(100vh-22rem))] 2xl:h-[min(42rem,calc(100vh-20rem))] xl:overflow-hidden">
				<div class="rounded border border-gray-800 p-3 flex flex-col flex-1 min-h-0">
					<div class="mb-2 shrink-0 space-y-2">
						<div class="flex items-center justify-between gap-3">
							<div class="flex items-center gap-1 text-xs font-mono">
								<button class="{listMode === 'leaderboard' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
									onclick={() => setListMode('leaderboard')}>leaderboard</button>
								<span class="text-gray-600">/</span>
								<button class="{listMode === 'current' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
									onclick={() => setListMode('current')}>history</button>
							</div>
							<div class="flex items-center gap-2 text-[10px] font-mono text-gray-500">
								<span>{experiments.length} dataset runs</span>
								<button
									onclick={() => {
										selectionMode = !selectionMode;
										if (!selectionMode) clearBatchSelection();
									}}
									disabled={running || importing}
									class="{selectionMode ? 'text-blue-300' : 'text-gray-500 hover:text-gray-300'} disabled:opacity-40"
								>
									{selectionMode ? 'done' : 'select'}
								</button>
							</div>
						</div>
						<p class="text-[10px] font-mono text-gray-600">
							showing {visibleLeaderboardExperiments.length} of {allExperiments.length} runs in {resultsScopeLabel}
						</p>
						<select
							bind:value={leaderboardSort}
							class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1 text-[10px] font-mono text-gray-400"
							title="sort experiments"
						>
							<option value="bpb">sort: {activeTrainer.metricShortLabel}</option>
							<option value="name">sort: a-z</option>
							<option value="newest">sort: newest</option>
							<option value="oldest">sort: oldest</option>
							<option value="steps">sort: steps</option>
						</select>
						<ResultsSummary
							experiments={experiments}
							scopeMode={resultsScope}
							metricShortLabel={activeDataset?.recipeKey?.includes('abc') ? 'abc' : activeTrainer.metricShortLabel}
							recipeKey={activeDataset?.recipeKey ?? null}
							selectedBenchmarkGroup={selectedBenchmarkGroup}
							selectedModelFamily={selectedModelFamily}
							onSelectAll={() => applyResultsScope('all')}
							onSelectAdHoc={() => applyResultsScope('adHoc')}
							onSelectReruns={() => applyResultsScope('reruns')}
							onSelectBenchmarkGroup={(group) => applyResultsScope('benchmark', group)}
							onSelectModelFamily={(family) => applyResultsScope('family', family)}
						/>
					</div>
					{#if selectionMode}
						<div class="mb-2 flex items-center gap-2 text-[10px] font-mono text-gray-500">
							<span>{selectedBatchIds.length} selected</span>
							<button onclick={rerunSelectedBatch} disabled={running || importing || selectedBatchIds.length === 0} class="text-gray-400 hover:text-white disabled:opacity-40">
								rerun selected
							</button>
							<button onclick={deleteSelectedBatch} disabled={running || importing || selectedBatchIds.length === 0} class="text-red-400 hover:text-red-200 disabled:opacity-40">
								delete selected
							</button>
							<button onclick={clearBatchSelection} disabled={selectedBatchIds.length === 0} class="text-gray-500 hover:text-gray-300 disabled:opacity-40">
								clear selection
							</button>
						</div>
					{/if}
					<div class="flex-1 min-h-0 overflow-hidden">
						<Leaderboard experiments={visibleLeaderboardExperiments} onSelect={selectExperiment}
							selected={selectedExp}
							sortMode={leaderboardSort}
							metricShortLabel={activeTrainer.metricShortLabel}
							extraColumns={activeMetricColumns}
							runNumberById={datasetRunNumberById}
							nextRunNumber={nextDatasetRunNumber}
							selectionEnabled={selectionMode}
							selectedIds={selectedBatchIds}
							onToggleBatchSelect={toggleBatchSelection} />
					</div>
					<div class="flex gap-3 mt-2 pt-2 border-t border-gray-800 shrink-0">
						<input bind:this={importInput} type="file" accept=".zip,application/zip" class="hidden" onchange={handleImportFile} />
						{#if experiments.length > 0}
							<button onclick={rerunAllExperiments} disabled={running || importing} class="text-gray-500 hover:text-gray-300 disabled:opacity-40 text-xs font-mono">rerun all</button>
							<button onclick={handleExport} class="text-gray-500 hover:text-gray-300 text-xs font-mono">export</button>
							<button onclick={handleClear} disabled={running} class="text-gray-500 hover:text-red-400 text-xs font-mono">clear</button>
						{/if}
					</div>
				</div>
			</div>
		</div>
		</div>
		{/if}
	{/if}

	{#if showClearModal}
		<div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
			<div class="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-sm space-y-4 font-mono">
				<h3 class="text-sm text-gray-200">clear all data?</h3>
				<p class="text-xs text-gray-400">this will delete all experiments, loss curves, inferences, and saved weights. this cannot be undone.</p>
				<div class="flex gap-2 justify-end">
					<button onclick={() => (showClearModal = false)} class="px-3 py-1.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm">cancel</button>
					<button onclick={confirmClear} class="px-3 py-1.5 rounded bg-red-600 text-white hover:bg-red-500 text-sm">clear everything</button>
				</div>
			</div>
		</div>
	{/if}
</main>
