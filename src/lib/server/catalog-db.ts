import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { DEFAULT_TRAINER_KEY } from '$lib/trainers';

const STORAGE_ROOT = join(process.cwd(), 'output', 'local');
const DATASET_ROOT = join(STORAGE_ROOT, 'datasets');
const DB_PATH = join(STORAGE_ROOT, 'catalog.sqlite');

let db: DatabaseSync | null = null;

const SCHEMA = `
	PRAGMA foreign_keys = ON;

	CREATE TABLE IF NOT EXISTS dataset_versions (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		source_type TEXT NOT NULL,
		source_ref TEXT NOT NULL,
		label TEXT NOT NULL,
		slug TEXT NOT NULL,
		config_name TEXT NOT NULL DEFAULT 'default',
		revision TEXT,
		recipe_key TEXT NOT NULL,
		trainer_key TEXT NOT NULL DEFAULT '${DEFAULT_TRAINER_KEY}',
		model_family TEXT NOT NULL DEFAULT 'byte-gpt',
		text_fields_json TEXT NOT NULL DEFAULT '[]',
		sample_prompt TEXT NOT NULL DEFAULT '',
		train_examples INTEGER NOT NULL DEFAULT 0,
		validation_examples INTEGER NOT NULL DEFAULT 0,
		train_bytes INTEGER NOT NULL DEFAULT 0,
		validation_bytes INTEGER NOT NULL DEFAULT 0,
		max_train_examples INTEGER,
		max_validation_examples INTEGER,
		manifest_path TEXT NOT NULL,
		train_path TEXT NOT NULL,
		validation_path TEXT NOT NULL,
		is_active INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_dataset_versions_active ON dataset_versions(is_active);
	CREATE INDEX IF NOT EXISTS idx_dataset_versions_source ON dataset_versions(source_ref, created_at DESC);

	CREATE TABLE IF NOT EXISTS experiments (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		project_id INTEGER,
		name TEXT NOT NULL DEFAULT '',
		source TEXT NOT NULL DEFAULT 'manual',
		code TEXT NOT NULL,
		dataset_version_id INTEGER,
		dataset_label TEXT,
		dataset_source_ref TEXT,
		trainer_key TEXT NOT NULL DEFAULT '${DEFAULT_TRAINER_KEY}',
		model_family TEXT NOT NULL DEFAULT 'byte-gpt',
		reasoning TEXT NOT NULL DEFAULT '',
		val_bpb REAL NOT NULL,
		primary_score REAL,
		elapsed REAL NOT NULL,
		total_steps INTEGER NOT NULL,
		kept INTEGER NOT NULL DEFAULT 0,
		error TEXT,
		loss_curve TEXT,
		eval_summary_json TEXT,
		weights_path TEXT,
		rerun_of INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
		benchmark_group TEXT,
		created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id) ON DELETE SET NULL,
		FOREIGN KEY (project_id) REFERENCES experiment_workspaces(id) ON DELETE SET NULL
	);

	CREATE TABLE IF NOT EXISTS experiment_workspaces (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		source_type TEXT NOT NULL,
		dataset_version_id INTEGER,
		base_experiment_id INTEGER,
		readme TEXT NOT NULL DEFAULT '',
		notes TEXT NOT NULL DEFAULT '',
		created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id) ON DELETE SET NULL,
		FOREIGN KEY (base_experiment_id) REFERENCES experiments(id) ON DELETE SET NULL
	);

	CREATE INDEX IF NOT EXISTS idx_experiment_workspaces_created ON experiment_workspaces(created_at DESC, id DESC);
	CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id);

	CREATE TABLE IF NOT EXISTS loss_steps (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		step INTEGER NOT NULL,
		loss REAL NOT NULL
	);

	CREATE INDEX IF NOT EXISTS idx_loss_steps_exp ON loss_steps(experiment_id);

	CREATE TABLE IF NOT EXISTS inferences (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		prompt TEXT NOT NULL DEFAULT '',
		output TEXT NOT NULL,
		temperature REAL NOT NULL DEFAULT 0.8,
		created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
`;

export type DatasetVersionRow = {
	id: number;
	source_type: 'huggingface' | 'legacy';
	source_ref: string;
	label: string;
	slug: string;
	config_name: string;
	revision: string | null;
	recipe_key: string;
	trainer_key: string;
	model_family: string;
	text_fields_json: string;
	sample_prompt: string;
	train_examples: number;
	validation_examples: number;
	train_bytes: number;
	validation_bytes: number;
	max_train_examples: number | null;
	max_validation_examples: number | null;
	manifest_path: string;
	train_path: string;
	validation_path: string;
	is_active: number;
	created_at: string;
};

function ensureStorage() {
	mkdirSync(DATASET_ROOT, { recursive: true });
	mkdirSync(dirname(DB_PATH), { recursive: true });
}

function hasColumn(conn: DatabaseSync, tableName: string, columnName: string): boolean {
	const columns = conn.prepare(`PRAGMA table_info(${tableName})`).all() as { name: string }[];
	return columns.some((column) => column.name === columnName);
}

function runMigrations(conn: DatabaseSync): void {
	if (!hasColumn(conn, 'dataset_versions', 'trainer_key')) {
		conn.exec(`ALTER TABLE dataset_versions ADD COLUMN trainer_key TEXT NOT NULL DEFAULT '${DEFAULT_TRAINER_KEY}'`);
	}
	if (!hasColumn(conn, 'dataset_versions', 'sample_prompt')) {
		conn.exec(`ALTER TABLE dataset_versions ADD COLUMN sample_prompt TEXT NOT NULL DEFAULT ''`);
	}
	if (!hasColumn(conn, 'experiments', 'trainer_key')) {
		conn.exec(`ALTER TABLE experiments ADD COLUMN trainer_key TEXT NOT NULL DEFAULT '${DEFAULT_TRAINER_KEY}'`);
	}
	if (!hasColumn(conn, 'experiments', 'project_id')) {
		conn.exec(`ALTER TABLE experiments ADD COLUMN project_id INTEGER`);
	}
	if (!hasColumn(conn, 'experiments', 'primary_score')) {
		conn.exec(`ALTER TABLE experiments ADD COLUMN primary_score REAL`);
	}
	if (!hasColumn(conn, 'experiments', 'eval_summary_json')) {
		conn.exec(`ALTER TABLE experiments ADD COLUMN eval_summary_json TEXT`);
	}
}

export function getStorageRoot(): string {
	ensureStorage();
	return STORAGE_ROOT;
}

export function getDatasetRoot(): string {
	ensureStorage();
	return DATASET_ROOT;
}

export function getCatalogDb(): DatabaseSync {
	if (db) return db;
	ensureStorage();
	db = new DatabaseSync(DB_PATH);
	db.exec(SCHEMA);
	runMigrations(db);
	return db;
}

export function listDatasetVersionRows(): DatasetVersionRow[] {
	const conn = getCatalogDb();
	return conn.prepare(`
		SELECT *
		FROM dataset_versions
		ORDER BY is_active DESC, created_at DESC, id DESC
	`).all() as DatasetVersionRow[];
}

export function getActiveDatasetVersionRow(): DatasetVersionRow | null {
	const conn = getCatalogDb();
	return conn.prepare(`
		SELECT *
		FROM dataset_versions
		WHERE is_active = 1
		ORDER BY id DESC
		LIMIT 1
	`).get() as DatasetVersionRow | undefined ?? null;
}

export function getDatasetVersionRowById(id: number): DatasetVersionRow | null {
	const conn = getCatalogDb();
	return conn.prepare(`
		SELECT *
		FROM dataset_versions
		WHERE id = ?
	`).get(id) as DatasetVersionRow | undefined ?? null;
}

export function insertDatasetVersion(row: Omit<DatasetVersionRow, 'id' | 'is_active' | 'created_at'> & { isActive?: boolean }): number {
	const conn = getCatalogDb();
	const result = conn.prepare(`
		INSERT INTO dataset_versions (
			source_type,
			source_ref,
			label,
			slug,
			config_name,
			revision,
			recipe_key,
			trainer_key,
			model_family,
			text_fields_json,
			sample_prompt,
			train_examples,
			validation_examples,
			train_bytes,
			validation_bytes,
			max_train_examples,
			max_validation_examples,
			manifest_path,
			train_path,
			validation_path,
			is_active
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`).run(
		row.source_type,
		row.source_ref,
		row.label,
		row.slug,
		row.config_name,
		row.revision,
		row.recipe_key,
		row.trainer_key,
		row.model_family,
		row.text_fields_json,
		row.sample_prompt,
		row.train_examples,
		row.validation_examples,
		row.train_bytes,
		row.validation_bytes,
		row.max_train_examples,
		row.max_validation_examples,
		row.manifest_path,
		row.train_path,
		row.validation_path,
		row.isActive ? 1 : 0
	);
	return Number(result.lastInsertRowid);
}

export function setActiveDatasetVersionRow(versionId: number): void {
	const conn = getCatalogDb();
	conn.exec('BEGIN');
	try {
		conn.prepare(`UPDATE dataset_versions SET is_active = 0 WHERE is_active != 0`).run();
		const result = conn.prepare(`UPDATE dataset_versions SET is_active = 1 WHERE id = ?`).run(versionId);
		if (result.changes === 0) {
			throw new Error(`Dataset version ${versionId} does not exist.`);
		}
		conn.exec('COMMIT');
	} catch (error) {
		conn.exec('ROLLBACK');
		throw error;
	}
}
