import { PGlite } from '@electric-sql/pglite';

let db: PGlite | null = null;

const SCHEMA = `
	CREATE TABLE IF NOT EXISTS experiments (
		id SERIAL PRIMARY KEY,
		name TEXT NOT NULL DEFAULT '',
		source TEXT NOT NULL DEFAULT 'manual',
		-- config columns
		n_layer INTEGER NOT NULL DEFAULT 3,
		n_embd INTEGER NOT NULL DEFAULT 96,
		n_head INTEGER NOT NULL DEFAULT 4,
		mlp_ratio INTEGER NOT NULL DEFAULT 4,
		activation TEXT NOT NULL DEFAULT 'relu_sq',
		use_rope BOOLEAN NOT NULL DEFAULT true,
		softcap_value REAL NOT NULL DEFAULT 15,
		lr REAL NOT NULL DEFAULT 0.001,
		weight_decay REAL NOT NULL DEFAULT 0.1,
		warmup_ratio REAL NOT NULL DEFAULT 0.1,
		cooldown_ratio REAL NOT NULL DEFAULT 0.3,
		batch_size INTEGER NOT NULL DEFAULT 8,
		seq_len INTEGER NOT NULL DEFAULT 128,
		train_seconds REAL NOT NULL DEFAULT 30,
		vocab_size INTEGER NOT NULL DEFAULT 256,
		-- results
		val_bpb REAL NOT NULL,
		elapsed REAL NOT NULL,
		total_steps INTEGER NOT NULL,
		reasoning TEXT NOT NULL DEFAULT '',
		kept BOOLEAN NOT NULL DEFAULT false,
		loss_curve JSONB,
		weights_path TEXT,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS loss_steps (
		id SERIAL PRIMARY KEY,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		step INTEGER NOT NULL,
		loss REAL NOT NULL
	);

	CREATE INDEX IF NOT EXISTS idx_loss_steps_exp ON loss_steps(experiment_id);

	CREATE TABLE IF NOT EXISTS inferences (
		id SERIAL PRIMARY KEY,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		prompt TEXT NOT NULL DEFAULT '',
		output TEXT NOT NULL,
		temperature REAL NOT NULL DEFAULT 0.8,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);
`;

export async function getDb(): Promise<PGlite> {
	if (db) return db;
	db = new PGlite('idb://autoresearch');
	await db.exec(SCHEMA);
	return db;
}

// -- Experiments --

export type ExperimentRow = {
	id: number;
	name: string;
	source: 'manual' | 'auto';
	n_layer: number;
	n_embd: number;
	n_head: number;
	mlp_ratio: number;
	activation: string;
	use_rope: boolean;
	softcap_value: number;
	lr: number;
	weight_decay: number;
	warmup_ratio: number;
	cooldown_ratio: number;
	batch_size: number;
	seq_len: number;
	train_seconds: number;
	vocab_size: number;
	val_bpb: number;
	elapsed: number;
	total_steps: number;
	reasoning: string;
	kept: boolean;
	loss_curve: { step: number; loss: number }[] | null;
	weights_path: string | null;
	created_at: string;
};

export async function insertExperiment(exp: {
	name?: string;
	source?: 'manual' | 'auto';
	config: Record<string, unknown>;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	lossCurve?: { step: number; loss: number }[];
}): Promise<number> {
	const pg = await getDb();
	const c = exp.config;
	const result = await pg.query<{ id: number }>(
		`INSERT INTO experiments (
			name, source,
			n_layer, n_embd, n_head, mlp_ratio, activation, use_rope, softcap_value,
			lr, weight_decay, warmup_ratio, cooldown_ratio, batch_size, seq_len, train_seconds, vocab_size,
			val_bpb, elapsed, total_steps, reasoning, kept, loss_curve
		) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)
		 RETURNING id`,
		[
			exp.name || '',
			exp.source || 'manual',
			c.nLayer, c.nEmbd, c.nHead, c.mlpRatio, c.activation, c.useRoPE, c.softcapValue,
			c.lr, c.weightDecay, c.warmupRatio, c.cooldownRatio, c.batchSize, c.seqLen, c.trainSeconds, c.vocabSize ?? 256,
			exp.valBpb,
			exp.elapsed,
			exp.totalSteps,
			exp.reasoning,
			exp.kept,
			exp.lossCurve ? JSON.stringify(exp.lossCurve) : null
		]
	);
	return result.rows[0].id;
}

/** Reconstruct an ExperimentConfig from a row's flat columns. */
export function rowToConfig(row: ExperimentRow) {
	return {
		nLayer: row.n_layer,
		nEmbd: row.n_embd,
		nHead: row.n_head,
		mlpRatio: row.mlp_ratio,
		activation: row.activation as 'relu_sq' | 'gelu' | 'silu',
		useRoPE: row.use_rope,
		softcapValue: row.softcap_value,
		lr: row.lr,
		weightDecay: row.weight_decay,
		warmupRatio: row.warmup_ratio,
		cooldownRatio: row.cooldown_ratio,
		batchSize: row.batch_size,
		seqLen: row.seq_len,
		trainSeconds: row.train_seconds,
		vocabSize: row.vocab_size,
	};
}

export async function getAllExperiments(): Promise<ExperimentRow[]> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(
		`SELECT * FROM experiments ORDER BY id`
	);
	return result.rows;
}

export async function getBestExperiment(): Promise<ExperimentRow | null> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(
		`SELECT * FROM experiments ORDER BY val_bpb ASC LIMIT 1`
	);
	return result.rows[0] ?? null;
}

export async function updateWeightsPath(id: number, weightsPath: string): Promise<void> {
	const pg = await getDb();
	await pg.query(`UPDATE experiments SET weights_path = $1 WHERE id = $2`, [weightsPath, id]);
}

export async function clearAllData(): Promise<void> {
	const pg = await getDb();
	await pg.exec(`
		DROP TABLE IF EXISTS inferences;
		DROP TABLE IF EXISTS loss_steps;
		DROP TABLE IF EXISTS experiments;
	`);
	await pg.exec(SCHEMA);
}

// -- Loss Steps --

export async function insertLossCurve(experimentId: number, curve: { step: number; loss: number }[]): Promise<void> {
	if (curve.length === 0) return;
	const pg = await getDb();
	const values = curve.map((_, i) => `($1, $${i * 2 + 2}, $${i * 2 + 3})`).join(',');
	const params: (number)[] = [experimentId];
	for (const point of curve) {
		params.push(point.step, point.loss);
	}
	await pg.query(`INSERT INTO loss_steps (experiment_id, step, loss) VALUES ${values}`, params);
}

export async function getLossCurve(experimentId: number): Promise<{ step: number; loss: number }[]> {
	const pg = await getDb();
	const result = await pg.query<{ step: number; loss: number }>(
		`SELECT step, loss FROM loss_steps WHERE experiment_id = $1 ORDER BY step`,
		[experimentId]
	);
	return result.rows;
}

export async function getAllLossCurves(): Promise<Map<number, { step: number; loss: number }[]>> {
	const pg = await getDb();
	const result = await pg.query<{ experiment_id: number; step: number; loss: number }>(
		`SELECT experiment_id, step, loss FROM loss_steps ORDER BY experiment_id, step`
	);
	const map = new Map<number, { step: number; loss: number }[]>();
	for (const row of result.rows) {
		if (!map.has(row.experiment_id)) map.set(row.experiment_id, []);
		map.get(row.experiment_id)!.push({ step: row.step, loss: row.loss });
	}
	return map;
}

// -- Inferences --

export type InferenceRow = {
	id: number;
	experiment_id: number;
	prompt: string;
	output: string;
	temperature: number;
	created_at: string;
};

export async function insertInference(inf: {
	experimentId: number;
	prompt: string;
	output: string;
	temperature: number;
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO inferences (experiment_id, prompt, output, temperature)
		 VALUES ($1, $2, $3, $4)
		 RETURNING id`,
		[inf.experimentId, inf.prompt, inf.output, inf.temperature]
	);
	return result.rows[0].id;
}

export async function getInferencesForExperiment(experimentId: number): Promise<InferenceRow[]> {
	const pg = await getDb();
	const result = await pg.query<InferenceRow>(
		`SELECT * FROM inferences WHERE experiment_id = $1 ORDER BY created_at DESC`,
		[experimentId]
	);
	return result.rows;
}

function toCsv(rows: Record<string, unknown>[]): string {
	if (rows.length === 0) return '';
	const keys = Object.keys(rows[0]);
	const escape = (v: unknown) => {
		const s = typeof v === 'object' ? JSON.stringify(v) : String(v ?? '');
		return s.includes(',') || s.includes('"') || s.includes('\n')
			? `"${s.replace(/"/g, '""')}"` : s;
	};
	const header = keys.join(',');
	const lines = rows.map(row => keys.map(k => escape(row[k])).join(','));
	return [header, ...lines].join('\n');
}

export async function exportCsvZip(): Promise<Blob> {
	const JSZip = (await import('jszip')).default;
	const pg = await getDb();

	const exps = await pg.query(
		`SELECT id, name, source,
			n_layer, n_embd, n_head, mlp_ratio, activation, use_rope, softcap_value,
			lr, weight_decay, warmup_ratio, cooldown_ratio, batch_size, seq_len, train_seconds, vocab_size,
			val_bpb, elapsed, total_steps, reasoning, kept, weights_path, created_at
		 FROM experiments ORDER BY id`
	);
	const steps = await pg.query(
		`SELECT id, experiment_id, step, loss FROM loss_steps ORDER BY experiment_id, step`
	);
	const infs = await pg.query(
		`SELECT id, experiment_id, prompt, output, temperature, created_at
		 FROM inferences ORDER BY experiment_id, created_at`
	);

	const zip = new JSZip();
	zip.file('experiments.csv', toCsv(exps.rows as Record<string, unknown>[]));
	zip.file('loss_steps.csv', toCsv(steps.rows as Record<string, unknown>[]));
	zip.file('inferences.csv', toCsv(infs.rows as Record<string, unknown>[]));

	return zip.generateAsync({ type: 'blob' });
}
