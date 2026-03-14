import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const cache = new Map<string, { data: unknown; ts: number }>();
const CACHE_TTL = 1000 * 60 * 60; // 1 hour
const MAX_CACHE = 200;

async function hashKey(system: string, user: string): Promise<string> {
	const data = new TextEncoder().encode(system + '\0' + user);
	const buf = await crypto.subtle.digest('SHA-256', data);
	return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
}

export const POST: RequestHandler = async ({ request, platform }) => {
	const apiKey = platform?.env?.ANTHROPIC_API_KEY;
	if (!apiKey) {
		return json({ error: 'API key not configured' }, { status: 500 });
	}

	// Cloudflare rate limiting
	const rateLimiter = platform?.env?.RATE_LIMITER;
	if (rateLimiter) {
		const { success } = await rateLimiter.limit({ key: 'global' });
		if (!success) {
			return json({ error: 'Rate limited. Max 1 request every 10 seconds.' }, { status: 429 });
		}
	}

	const { systemPrompt, userPrompt } = await request.json();

	// Check cache
	const key = await hashKey(systemPrompt, userPrompt);
	const cached = cache.get(key);
	if (cached && Date.now() - cached.ts < CACHE_TTL) {
		return json(cached.data);
	}

	const response = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-api-key': apiKey,
			'anthropic-version': '2023-06-01'
		},
		body: JSON.stringify({
			model: 'claude-sonnet-4-6',
			max_tokens: 1024,
			system: systemPrompt,
			messages: [{ role: 'user', content: userPrompt }]
		})
	});

	if (!response.ok) {
		const error = await response.text();
		return json({ error }, { status: response.status });
	}

	const data = await response.json();
	const text = data.content[0].text;

	const jsonMatch = text.match(/\{[\s\S]*\}/);
	if (!jsonMatch) {
		return json({ error: 'No JSON found in response', raw: text }, { status: 422 });
	}

	try {
		const parsed = JSON.parse(jsonMatch[0]);

		// Store in cache (evict oldest if full)
		if (cache.size >= MAX_CACHE) {
			const oldest = cache.keys().next().value!;
			cache.delete(oldest);
		}
		cache.set(key, { data: parsed, ts: Date.now() });

		return json(parsed);
	} catch {
		return json({ error: 'Invalid JSON in response', raw: text }, { status: 422 });
	}
};
