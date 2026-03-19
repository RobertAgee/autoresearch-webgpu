import { mkdirSync } from 'node:fs';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const ARTIFACT_ROOT = join(process.cwd(), 'output', 'local', 'artifacts');
const WEIGHTS_ROOT = join(ARTIFACT_ROOT, 'weights');

export type WeightMeta = {
	key: string;
	shape: number[];
	dtype: string;
	offset: number;
	byteLength: number;
};

function ensureArtifactsRoot(): void {
	mkdirSync(WEIGHTS_ROOT, { recursive: true });
}

function weightDataPath(experimentId: number): string {
	return join(WEIGHTS_ROOT, `exp-${experimentId}.bin`);
}

function weightMetaPath(experimentId: number): string {
	return join(WEIGHTS_ROOT, `exp-${experimentId}.meta.json`);
}

export function getArtifactPath(relativePath: string): string {
	ensureArtifactsRoot();
	return join(ARTIFACT_ROOT, relativePath);
}

export async function saveWeightsArtifact(
	experimentId: number,
	buffer: Uint8Array,
	metas: WeightMeta[]
): Promise<string> {
	ensureArtifactsRoot();
	const dataPath = weightDataPath(experimentId);
	const metaPath = weightMetaPath(experimentId);
	await writeFile(dataPath, buffer);
	await writeFile(metaPath, JSON.stringify(metas));
	return `weights/exp-${experimentId}.bin`;
}

export async function loadWeightsArtifact(experimentId: number): Promise<{ buffer: Uint8Array; metas: WeightMeta[] } | null> {
	try {
		ensureArtifactsRoot();
		const [buffer, metaText] = await Promise.all([
			readFile(weightDataPath(experimentId)),
			readFile(weightMetaPath(experimentId), 'utf8')
		]);
		return {
			buffer: new Uint8Array(buffer),
			metas: JSON.parse(metaText) as WeightMeta[]
		};
	} catch {
		return null;
	}
}

export async function deleteWeightsArtifact(experimentId: number): Promise<void> {
	await Promise.all([
		rm(weightDataPath(experimentId), { force: true }),
		rm(weightMetaPath(experimentId), { force: true })
	]);
}

export async function clearWeightsArtifacts(): Promise<void> {
	await rm(WEIGHTS_ROOT, { recursive: true, force: true });
	mkdirSync(dirname(weightDataPath(0)), { recursive: true });
}
