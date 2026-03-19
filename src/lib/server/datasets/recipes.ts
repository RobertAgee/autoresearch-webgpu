export type DatasetRecipe = {
	key: string;
	trainerKey: string;
	modelFamily: string;
	description: string;
	preprocessingSummary: string;
	preprocessingSteps: string[];
	researchNotes: string[];
	textFields: string[];
	samplePrompt: string;
	renderExample: (row: Record<string, unknown>) => string;
};

export type DatasetRecipeOption = {
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
};

function asString(value: unknown): string {
	return typeof value === 'string' ? value.trim() : '';
}

function normalizeFieldName(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/(^-|-$)/g, '') || 'field';
}

function wrapSection(name: string, value: string): string {
	const trimmed = value.trim();
	if (!trimmed) return '';
	return [`[${name}]`, trimmed, `[/${name}]`].join('\n');
}

const LEGACY_BYTE_BIN_RECIPE: DatasetRecipe = {
	key: 'legacy-byte-bin-v1',
	trainerKey: 'byte-next-token-v1',
	modelFamily: 'byte-gpt',
	description: 'Legacy byte-level corpus bundled with the repo.',
	preprocessingSummary: 'Uses the prebuilt train.bin and val.bin files as-is with no row-level transforms.',
	preprocessingSteps: [
		'Read the existing binary train and validation artifacts directly.',
		'Do not assume field boundaries or structured sections exist in the source bytes.'
	],
	researchNotes: [
		'Treat this as an opaque byte corpus because the original row structure is unavailable.',
		'Architecture and optimization changes are safer than assumptions about document formatting.'
	],
	textFields: [],
	samplePrompt: '',
	renderExample() {
		return '';
	}
};

const IRISHMAN_RECIPE_V1: DatasetRecipe = {
	key: 'abc-control-v1',
	trainerKey: 'byte-next-token-v1',
	modelFamily: 'byte-gpt',
	description: 'Simple conditioning on control code followed by ABC notation.',
	preprocessingSummary: 'Trim both text fields and concatenate control code with ABC notation separated by a newline.',
	preprocessingSteps: [
		'Trim the control code field.',
		'Trim the abc notation field.',
		'Join the non-empty fields with a single newline.'
	],
	researchNotes: [
		'This encoding is weakly structured, so the model must infer field boundaries from plain text.',
		'Loss is useful, but generated ABC validity and repetition matter more than tiny bpb gains.'
	],
	textFields: ['control code', 'abc notation'],
	samplePrompt: '',
	renderExample(row) {
		const control = asString(row['control code']);
		const notation = asString(row['abc notation']);
		return [control, notation].filter(Boolean).join('\n');
	}
};

const IRISHMAN_RECIPE_V2: DatasetRecipe = {
	key: 'abc-control-v2',
	trainerKey: 'byte-next-token-v1',
	modelFamily: 'byte-gpt',
	description: 'Structured ABC music conditioning with explicit control and notation sections.',
	preprocessingSummary: 'Wrap control metadata and ABC notation in stable section markers so the byte-level model can learn boundaries explicitly.',
	preprocessingSteps: [
		'Trim the control code field.',
		'Trim the abc notation field.',
		'Emit [control] ... [/control] followed by [abc] ... [/abc] for each example.',
		'Drop rows that do not contain ABC notation after trimming.'
	],
	researchNotes: [
		'This is symbolic music, not prose. Favor stable local syntax and long-range musical structure over generic text fluency.',
		'Do not strip or rewrite ABC symbols in-model; treat barlines, headers, note lengths, and repeats as meaningful bytes.',
		'When evaluating candidate code, prefer changes that reduce collapse, endless repetition, and malformed ABC output.'
	],
	textFields: ['control code', 'abc notation'],
	samplePrompt: '[control]\n',
	renderExample(row) {
		const control = asString(row['control code']);
		const notation = asString(row['abc notation']);
		if (!notation) return '';
		return [
			wrapSection('control', control || 'unspecified'),
			wrapSection('abc', notation)
		].filter(Boolean).join('\n\n');
	}
};

const GENERIC_STRING_RECIPE_V1: DatasetRecipe = {
	key: 'all-string-fields-v1',
	trainerKey: 'byte-next-token-v1',
	modelFamily: 'byte-gpt',
	description: 'Generic string field concatenation with field labels.',
	preprocessingSummary: 'Serialize each string field as "field name: value" blocks separated by blank lines.',
	preprocessingSteps: [
		'Keep every string-valued field.',
		'Trim each value.',
		'Prefix each value with its field name and separate fields with blank lines.'
	],
	researchNotes: [
		'This recipe preserves broad field context but does not enforce strong structure.',
		'If multiple fields exist, models may benefit from narrower context windows and simpler architectures.'
	],
	textFields: [],
	samplePrompt: '',
	renderExample(row) {
		const text = Object.entries(row)
			.filter(([, value]) => typeof value === 'string')
			.map(([key, value]) => `${key}:\n${String(value).trim()}`)
			.filter((value) => value.length > 0)
			.join('\n\n');
		return text;
	}
};

function buildGenericStringRecipeV2(fields: string[]): DatasetRecipe {
	const selectedFields = fields.length > 0 ? fields : [];
	const multiField = selectedFields.length > 1;

	return {
		key: 'all-string-fields-v2',
		trainerKey: 'byte-next-token-v1',
		modelFamily: 'byte-gpt',
		description: multiField
			? 'Structured multi-field text packing with explicit section markers.'
			: 'Single-field text packing with minimal extra framing.',
		preprocessingSummary: multiField
			? 'Trim string fields and wrap each field in [field-name] markers so boundaries survive byte-level training.'
			: 'Trim the single dominant text field and keep the payload mostly raw to avoid wasting context on labels.',
		preprocessingSteps: multiField
			? [
				'Keep the configured string fields in a stable order.',
				'Trim each field value.',
				'Wrap each non-empty field in [field-name] ... [/field-name] markers.',
				'Join field sections with blank lines.'
			]
			: [
				'Keep the configured text field.',
				'Trim leading and trailing whitespace.',
				'Store the raw payload without extra field headers when possible.'
			],
		researchNotes: multiField
			? [
				'Respect field boundaries. Architectural changes that blur sections may hurt more than they help.',
				'If one field dominates length, bias experiments toward shorter context or stronger regularization.'
			]
			: [
				'Treat this as plain sequence modeling over one primary text field.',
				'Model capacity and optimization matter more than field serialization because structure is minimal.'
			],
		textFields: selectedFields,
		samplePrompt: '',
		renderExample(row) {
			const availableFields = selectedFields.length > 0
				? selectedFields
				: Object.entries(row)
					.filter(([, value]) => typeof value === 'string')
					.map(([key]) => key);

			if (availableFields.length === 1) {
				return asString(row[availableFields[0]]);
			}

			return availableFields
				.map((field) => wrapSection(normalizeFieldName(field), asString(row[field])))
				.filter(Boolean)
				.join('\n\n');
		}
	};
}

const RECIPE_BY_KEY: Record<string, DatasetRecipe> = {
	[LEGACY_BYTE_BIN_RECIPE.key]: LEGACY_BYTE_BIN_RECIPE,
	[IRISHMAN_RECIPE_V1.key]: IRISHMAN_RECIPE_V1,
	[IRISHMAN_RECIPE_V2.key]: IRISHMAN_RECIPE_V2,
	[GENERIC_STRING_RECIPE_V1.key]: GENERIC_STRING_RECIPE_V1
};

export function listApplicableDatasetRecipes(
	sourceRef: string,
	fields: string[]
): DatasetRecipeOption[] {
	const fromRecipe = (label: string, recipe: DatasetRecipe): DatasetRecipeOption => ({
		key: recipe.key,
		label,
		description: recipe.description,
		trainerKey: recipe.trainerKey,
		modelFamily: recipe.modelFamily,
		textFields: recipe.textFields,
		preprocessingSummary: recipe.preprocessingSummary,
		preprocessingSteps: recipe.preprocessingSteps,
		researchNotes: recipe.researchNotes,
		samplePrompt: recipe.samplePrompt
	});

	if (sourceRef === 'sander-wood/irishman') {
		const genericStructured = buildGenericStringRecipeV2(fields);
		return [
			fromRecipe('Irishman structured', IRISHMAN_RECIPE_V2),
			fromRecipe('Irishman simple join', IRISHMAN_RECIPE_V1),
			fromRecipe('Generic structured text', genericStructured),
			fromRecipe('Generic labeled text', {
				...GENERIC_STRING_RECIPE_V1,
				textFields: fields
			})
		];
	}

	if (fields.length > 0) {
		const genericStructured = buildGenericStringRecipeV2(fields);
		return [
			fromRecipe('Structured text', genericStructured),
			fromRecipe('Labeled text', {
				...GENERIC_STRING_RECIPE_V1,
				textFields: fields
			})
		];
	}

	return [
		fromRecipe('Labeled text', GENERIC_STRING_RECIPE_V1)
	];
}

export function resolveDatasetRecipe(
	sourceRef: string,
	fields: string[],
	recipeKey?: string | null
): DatasetRecipe {
	if (recipeKey === 'all-string-fields-v2') {
		return buildGenericStringRecipeV2(fields);
	}

	if (recipeKey && RECIPE_BY_KEY[recipeKey]) {
		const recipe = RECIPE_BY_KEY[recipeKey];
		return recipe.textFields.length > 0 || fields.length === 0
			? recipe
			: { ...recipe, textFields: fields };
	}

	if (sourceRef === 'sander-wood/irishman') {
		return IRISHMAN_RECIPE_V2;
	}

	if (fields.length > 0) {
		return buildGenericStringRecipeV2(fields);
	}

	return {
		...GENERIC_STRING_RECIPE_V1,
		textFields: fields
	};
}
