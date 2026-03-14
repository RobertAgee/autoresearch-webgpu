import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

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

	const response = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-api-key': apiKey,
			'anthropic-version': '2023-06-01',
		},
		body: JSON.stringify({
			model: 'claude-sonnet-4-6',
			max_tokens: 8192,
			// Prompt caching: system prompt is large (API reference) and stable across calls.
			// Using cache_control on the system block tells Anthropic to cache it.
			system: [
				{
					type: 'text',
					text: systemPrompt,
					cache_control: { type: 'ephemeral' }
				}
			],
			messages: [{ role: 'user', content: userPrompt }]
		})
	});

	if (!response.ok) {
		const error = await response.text();
		return json({ error }, { status: response.status });
	}

	const data = await response.json();
	const text = data.content[0].text;

	// Try to parse the JSON response. Claude should return { reasoning, code }.
	// The code field contains braces, so we can't use a simple regex.
	// Strategy: try JSON.parse on the full text first, then look for JSON block.
	try {
		const parsed = JSON.parse(text);
		if (parsed.code && parsed.reasoning) {
			return json(parsed);
		}
	} catch {
		// Not direct JSON, try extracting from markdown fences or finding the JSON object
	}

	// Try to find JSON between ```json ... ``` or ``` ... ```
	const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenceMatch) {
		try {
			const parsed = JSON.parse(fenceMatch[1].trim());
			if (parsed.code) return json(parsed);
		} catch {}
	}

	// Last resort: find the outermost { ... } but be smarter about nested braces
	try {
		const start = text.indexOf('{');
		if (start >= 0) {
			let depth = 0;
			let end = start;
			for (let i = start; i < text.length; i++) {
				if (text[i] === '{') depth++;
				else if (text[i] === '}') {
					depth--;
					if (depth === 0) { end = i; break; }
				}
			}
			const parsed = JSON.parse(text.slice(start, end + 1));
			if (parsed.code) return json(parsed);
		}
	} catch {}

	return json({ error: 'Could not parse response', raw: text.slice(0, 500) }, { status: 422 });
};
