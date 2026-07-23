import type { EvalJob, EvalRun, ModelOption, RunSummary, StartEvalPayload } from './types'

async function readError(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string | { msg?: string }[] }
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg ?? JSON.stringify(d)).join('; ')
    }
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const res = await fetch('/api/runs')
  if (!res.ok) throw new Error(await readError(res))
  const data = (await res.json()) as { runs: RunSummary[] }
  return data.runs
}

export async function fetchRun(runId: string): Promise<EvalRun> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as EvalRun
}

export async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await readError(res))
}

export async function fetchModels(): Promise<{ models: ModelOption[]; default: string }> {
  const res = await fetch('/api/models')
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as { models: ModelOption[]; default: string }
}

export async function startEval(payload: StartEvalPayload): Promise<EvalJob> {
  const res = await fetch('/api/evals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await readError(res))
  const data = (await res.json()) as { job: EvalJob }
  return data.job
}

export async function fetchJob(jobId: string): Promise<EvalJob> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`)
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as EvalJob
}
