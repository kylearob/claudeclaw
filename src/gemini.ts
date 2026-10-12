import os from 'os';

import { query } from '@anthropic-ai/claude-agent-sdk';
import { GoogleGenAI } from '@google/genai';

import { buildSdkEnv, claudeExecutableOption } from './agent.js';
import { GOOGLE_API_KEY, MEMORY_LLM, MEMORY_MODEL } from './config.js';
import { logger } from './logger.js';

/** True when a memory model backend is configured and usable. */
export function memoryLlmEnabled(): boolean {
  if (MEMORY_LLM === 'off') return false;
  if (MEMORY_LLM === 'gemini') return !!GOOGLE_API_KEY;
  return true;
}

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  if (client) return client;
  if (!GOOGLE_API_KEY) {
    throw new Error('GOOGLE_API_KEY is not set. Add it to .env for memory extraction.');
  }
  client = new GoogleGenAI({ apiKey: GOOGLE_API_KEY });
  return client;
}

/**
 * Generate a JSON text response from the configured memory model
 * (MEMORY_LLM: Claude via the Agent SDK, or Gemini).
 */
export async function generateContent(prompt: string): Promise<string> {
  if (MEMORY_LLM === 'off') throw new Error('MEMORY_LLM is off');
  if (MEMORY_LLM === 'gemini') return generateContentGemini(prompt);
  return generateContentClaude(prompt);
}

// ── Claude backend ──────────────────────────────────────────────────

const CLAUDE_SYSTEM_PROMPT =
  'You are a background JSON processor for a personal assistant\'s memory system. ' +
  'Respond with only the JSON the prompt asks for. No prose, no code fences.';

const CLAUDE_TIMEOUT_MS = 90_000;

// One memory call at a time: each call spawns a Claude Code subprocess, and
// fire-and-forget ingestion must not pile them up on a small machine.
let claudeChain: Promise<unknown> = Promise.resolve();

function generateContentClaude(prompt: string): Promise<string> {
  const run = claudeChain.then(() => runClaudeOnce(prompt));
  claudeChain = run.catch(() => {});
  return run;
}

async function runClaudeOnce(prompt: string): Promise<string> {
  const abortController = new AbortController();
  const timer = setTimeout(() => abortController.abort(), CLAUDE_TIMEOUT_MS);
  let text = '';
  try {
    for await (const event of query({
      prompt,
      options: {
        model: MEMORY_MODEL,
        systemPrompt: CLAUDE_SYSTEM_PROMPT,
        // Bare call: no CLAUDE.md, skills, settings, tools, or MCP servers,
        // so each request carries only this prompt.
        settingSources: [],
        tools: [],
        mcpServers: {},
        strictMcpConfig: true,
        maxTurns: 1,
        thinking: { type: 'disabled' },
        persistSession: false,
        cwd: os.tmpdir(),
        env: buildSdkEnv(),
        abortController,
        ...claudeExecutableOption(),
      },
    })) {
      const ev = event as Record<string, unknown>;
      if (ev['type'] === 'result') {
        if (ev['subtype'] !== 'success') {
          throw new Error(`Memory model call failed: ${String(ev['subtype'])}`);
        }
        text = (ev['result'] as string | undefined) ?? '';
      }
    }
  } finally {
    clearTimeout(timer);
  }
  return text;
}

// ── Gemini backend ──────────────────────────────────────────────────

async function generateContentGemini(
  prompt: string,
  model = 'gemini-3-flash-preview',
): Promise<string> {
  const ai = getClient();
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      temperature: 0.1,
      responseMimeType: 'application/json',
    },
  });
  return response.text ?? '';
}

/**
 * Parse a JSON response from the memory model, with fallback on malformed output.
 * Returns null if parsing fails.
 */
export function parseJsonResponse<T>(text: string): T | null {
  try {
    // Strip markdown code fences if present
    const cleaned = text
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();
    try {
      return JSON.parse(cleaned) as T;
    } catch (err) {
      // Claude sometimes wraps JSON in a sentence; take the outermost JSON value
      const match = cleaned.match(/[[{][\s\S]*[\]}]/);
      if (!match) throw err;
      return JSON.parse(match[0]) as T;
    }
  } catch (err) {
    logger.warn({ err, text: text.slice(0, 200) }, 'Failed to parse memory model JSON response');
    return null;
  }
}
