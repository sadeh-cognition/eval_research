import { useCallback, useEffect, useMemo, useState } from 'react'
import { deleteRun, fetchJob, fetchModels, fetchRun, fetchRuns, startEval } from './api'
import {
  formatAggregateScore,
  formatExampleScore,
  formatTime,
  scoreClass,
} from './format'
import type {
  EvalJob,
  EvalRun,
  ExampleResult,
  LlmCall,
  ModelOption,
  RunSummary,
  StartEvalPayload,
} from './types'
import './App.css'

type FilterMode = 'all' | 'perfect' | 'partial' | 'failed'

function matchesFilter(score: number | null | undefined, mode: FilterMode): boolean {
  const s = score ?? 0
  if (mode === 'perfect') return s >= 1
  if (mode === 'partial') return s > 0 && s < 1
  if (mode === 'failed') return s === 0
  return true
}

function TrajectoryView({ trajectory }: { trajectory: ExampleResult['trajectory'] }) {
  if (!trajectory || typeof trajectory !== 'object' || Array.isArray(trajectory)) {
    return <pre className="block">{JSON.stringify(trajectory, null, 2)}</pre>
  }
  const traj = trajectory as Record<string, unknown>
  const stepIdxs = Array.from(
    new Set(
      Object.keys(traj)
        .filter((k) => /_\d+$/.test(k))
        .map((k) => Number(k.split('_').pop())),
    ),
  ).sort((a, b) => a - b)

  if (stepIdxs.length === 0) {
    return <pre className="block">{JSON.stringify(traj, null, 2)}</pre>
  }

  return (
    <div className="trajectory">
      {stepIdxs.map((s) => (
        <div key={s} className="traj-step">
          <div className="traj-step-title">Step {s}</div>
          {(['thought', 'tool_name', 'tool_args', 'observation'] as const).map((field) => {
            const key = `${field}_${s}`
            if (!(key in traj)) return null
            const val = traj[key]
            return (
              <div key={key} className="traj-field">
                <code>{key}</code>
                <pre className="block">
                  {typeof val === 'string' ? val.slice(0, 2000) : JSON.stringify(val, null, 2)}
                </pre>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

function LlmCallsView({ calls }: { calls: LlmCall[] }) {
  if (!calls.length) {
    return <p className="muted small">No LLM calls recorded for this example.</p>
  }
  return (
    <div className="llm-calls">
      {calls.map((call) => (
        <details key={call.uuid ?? call.call_index} className="llm-call" open={calls.length <= 3}>
          <summary>
            Call #{call.call_index + 1}
            {call.model ? ` · ${call.model}` : ''}
            {call.timestamp ? ` · ${call.timestamp.slice(0, 19).replace('T', ' ')}` : ''}
            {call.usage && typeof call.usage.total_tokens === 'number'
              ? ` · ${call.usage.total_tokens} tokens`
              : ''}
          </summary>
          <div className="llm-call-body">
            {(call.messages ?? []).map((msg, i) => (
              <div key={i} className={`llm-msg role-${msg.role}`}>
                <div className="llm-msg-role">{msg.role}</div>
                <pre className="block">{msg.content}</pre>
              </div>
            ))}
            {!call.messages?.length && call.prompt ? (
              <div className="llm-msg role-user">
                <div className="llm-msg-role">prompt</div>
                <pre className="block">{call.prompt}</pre>
              </div>
            ) : null}
            <div className="llm-msg role-assistant">
              <div className="llm-msg-role">assistant output</div>
              <pre className="block">
                {call.response_text || (call.outputs ?? []).join('\n') || '—'}
              </pre>
            </div>
            {call.usage && Object.keys(call.usage).length > 0 ? (
              <p className="muted small">usage: {JSON.stringify(call.usage)}</p>
            ) : null}
          </div>
        </details>
      ))}
    </div>
  )
}

function ExampleDetail({ index, row }: { index: number; row: ExampleResult }) {
  const gold = row.gold_titles ?? []
  const pred = row.pred_titles ?? []
  const predTop5 = new Set(pred.slice(0, 5))
  const goldSet = new Set(gold)
  const hits = gold.filter((t) => predTop5.has(t))
  const misses = gold.filter((t) => !predTop5.has(t))
  const extra = pred.slice(0, 5).filter((t) => !goldSet.has(t))
  const llmCalls = row.llm_calls ?? []

  return (
    <section className="example-detail">
      <h3>
        Example #{index}{' '}
        <span className={scoreClass(row.score)}>{formatExampleScore(row.score)}</span>
        {row.n_llm_calls != null ? (
          <span className="muted small"> · {row.n_llm_calls} LLM call(s)</span>
        ) : null}
      </h3>
      <div className="claim">{row.claim || '—'}</div>
      {row.error ? <div className="error">Example error: {row.error}</div> : null}
      <div className="title-cols">
        <div>
          <h4>Gold titles</h4>
          <ul>
            {gold.map((t) => (
              <li key={t}>
                {predTop5.has(t) ? '✅' : '❌'} <code>{t}</code>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Predicted titles</h4>
          <ul>
            {pred.map((t, i) => (
              <li key={`${t}-${i}`}>
                {goldSet.has(t) ? '✅' : '·'} {i + 1}. <code>{t}</code>
                {i < 5 ? <span className="muted"> (top-5)</span> : null}
              </li>
            ))}
          </ul>
        </div>
      </div>
      <p className="muted small">
        Hits: {hits.join(', ') || '—'} · Misses: {misses.join(', ') || '—'} · Extra in top-5:{' '}
        {extra.join(', ') || '—'}
      </p>
      {row.reasoning ? (
        <details>
          <summary>Reasoning</summary>
          <pre className="block">{row.reasoning}</pre>
        </details>
      ) : null}
      {row.trajectory ? (
        <details>
          <summary>Trajectory</summary>
          <TrajectoryView trajectory={row.trajectory} />
        </details>
      ) : null}
      <details open>
        <summary>LLM responses ({llmCalls.length})</summary>
        <LlmCallsView calls={llmCalls} />
      </details>
    </section>
  )
}

function RunEvalPanel({
  onCompleted,
}: {
  onCompleted: (runId: string) => void
}) {
  const [models, setModels] = useState<ModelOption[]>([])
  const [studentLm, setStudentLm] = useState('openai/gpt-5.4-mini')
  const [backend, setBackend] = useState<StartEvalPayload['backend']>('wikipedia')
  const [devSize, setDevSize] = useState(5)
  const [trainSize, setTrainSize] = useState(5)
  const [maxIters, setMaxIters] = useState(10)
  const [submitting, setSubmitting] = useState(false)
  const [job, setJob] = useState<EvalJob | null>(null)
  const [formError, setFormError] = useState<string | null>(null)

  useEffect(() => {
    void fetchModels()
      .then((data) => {
        setModels(data.models)
        setStudentLm(data.default || data.models[0]?.id || 'openai/gpt-5.4-mini')
      })
      .catch((e: unknown) => {
        setFormError(e instanceof Error ? e.message : String(e))
      })
  }, [])

  // Poll active job until terminal state.
  useEffect(() => {
    if (!job || job.status === 'succeeded' || job.status === 'failed') return
    const timer = window.setInterval(() => {
      void fetchJob(job.id)
        .then((next) => {
          setJob(next)
          if (next.status === 'succeeded' && next.run_id) {
            onCompleted(next.run_id)
          }
        })
        .catch((e: unknown) => {
          setFormError(e instanceof Error ? e.message : String(e))
        })
    }, 2000)
    return () => window.clearInterval(timer)
  }, [job, onCompleted])

  const busy = submitting || job?.status === 'queued' || job?.status === 'running'

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setFormError(null)
    setSubmitting(true)
    try {
      const payload: StartEvalPayload = {
        student_lm: studentLm,
        backend,
        train_size: trainSize,
        dev_size: devSize,
        max_iters: maxIters,
        num_threads: 2,
        safe: true,
        temperature: 0.7,
      }
      const started = await startEval(payload)
      setJob(started)
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="run-eval">
      <h3>Run evaluation</h3>
      <p className="muted small">
        Starts a HoVer <code>top5_recall</code> eval and appends results to disk history.
      </p>
      <form className="run-form" onSubmit={(e) => void handleSubmit(e)}>
        <label className="field">
          <span>Model</span>
          <select
            value={studentLm}
            onChange={(e) => setStudentLm(e.target.value)}
            disabled={busy || models.length === 0}
            required
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} ({m.id})
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Search backend</span>
          <select
            value={backend}
            onChange={(e) => setBackend(e.target.value as StartEvalPayload['backend'])}
            disabled={busy}
          >
            <option value="wikipedia">wikipedia</option>
            <option value="auto">auto</option>
            <option value="colbert">colbert</option>
          </select>
        </label>

        <div className="field-row">
          <label className="field">
            <span>Dev size</span>
            <input
              type="number"
              min={1}
              max={200}
              value={devSize}
              disabled={busy}
              onChange={(e) => setDevSize(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Train size</span>
            <input
              type="number"
              min={1}
              max={200}
              value={trainSize}
              disabled={busy}
              onChange={(e) => setTrainSize(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Max iters</span>
            <input
              type="number"
              min={1}
              max={30}
              value={maxIters}
              disabled={busy}
              onChange={(e) => setMaxIters(Number(e.target.value))}
            />
          </label>
        </div>

        <button type="submit" className="btn primary" disabled={busy || !studentLm}>
          {busy ? 'Running…' : 'Start eval'}
        </button>
      </form>

      {formError ? <div className="error">{formError}</div> : null}

      {job ? (
        <div className={`job-status status-${job.status}`}>
          <div>
            <strong>Job</strong> <code>{job.id}</code> · <span className="badge">{job.status}</span>
          </div>
          {job.status === 'succeeded' ? (
            <div className="muted small">
              Score {formatAggregateScore(job.score ?? null)} · run{' '}
              <code>{job.run_id}</code>
            </div>
          ) : null}
          {job.status === 'failed' && job.error ? (
            <pre className="block error-block">{job.error}</pre>
          ) : null}
          {(job.status === 'queued' || job.status === 'running') && (
            <p className="muted small">Polling every 2s… keep this tab open.</p>
          )}
        </div>
      ) : null}
    </section>
  )
}

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [run, setRun] = useState<EvalRun | null>(null)
  const [filter, setFilter] = useState<FilterMode>('all')
  const [exampleIdx, setExampleIdx] = useState(0)
  const [loadingList, setLoadingList] = useState(true)
  const [loadingRun, setLoadingRun] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadList = useCallback(async (preferId?: string | null) => {
    setLoadingList(true)
    setError(null)
    try {
      const data = await fetchRuns()
      setRuns(data)
      setSelectedId((prev) => {
        if (preferId === null) return data[0]?.id ?? null
        if (preferId) return preferId
        if (prev && data.some((r) => r.id === prev)) return prev
        return data[0]?.id ?? null
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingList(false)
    }
  }, [])

  useEffect(() => {
    void loadList()
  }, [loadList])

  const handleEvalCompleted = useCallback(
    (runId: string) => {
      void loadList(runId)
    },
    [loadList],
  )

  async function handleDeleteRun() {
    if (!selectedId || deleting) return
    const ok = window.confirm(
      `Delete eval run "${selectedId}"?\n\nThis permanently removes the file from disk.`,
    )
    if (!ok) return
    setDeleting(true)
    setError(null)
    try {
      await deleteRun(selectedId)
      setRun(null)
      await loadList(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setDeleting(false)
    }
  }

  useEffect(() => {
    if (!selectedId) {
      setRun(null)
      return
    }
    let cancelled = false
    setLoadingRun(true)
    setError(null)
    void fetchRun(selectedId)
      .then((data) => {
        if (!cancelled) {
          setRun(data)
          setExampleIdx(0)
          setFilter('all')
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoadingRun(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const results = run?.results ?? []
  const filtered = useMemo(
    () =>
      results
        .map((row, index) => ({ row, index }))
        .filter(({ row }) => matchesFilter(row.score, filter)),
    [results, filter],
  )

  useEffect(() => {
    if (filtered.length === 0) return
    if (!filtered.some((f) => f.index === exampleIdx)) {
      setExampleIdx(filtered[0].index)
    }
  }, [filtered, exampleIdx])

  const selectedExample = results[exampleIdx]

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-head">
          <h1>Eval history</h1>
          <button type="button" className="btn" onClick={() => void loadList()} disabled={loadingList}>
            Refresh
          </button>
        </div>
        <p className="muted small">Persisted runs under artifacts/evals</p>

        <RunEvalPanel onCompleted={handleEvalCompleted} />

        {loadingList ? <p className="muted">Loading…</p> : null}
        {!loadingList && runs.length === 0 ? (
          <p className="muted">No runs yet. Start an eval above.</p>
        ) : null}
        <ul className="run-list">
          {runs.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                className={r.id === selectedId ? 'run-item active' : 'run-item'}
                onClick={() => setSelectedId(r.id)}
              >
                <div className="run-item-top">
                  <span className={scoreClass(r.score)}>{formatAggregateScore(r.score)}</span>
                  <span className="badge">{r.kind}</span>
                </div>
                <div className="run-item-meta">{formatTime(r.created_at)}</div>
                <div className="run-item-meta">
                  n={r.n_examples} · {r.student_lm ?? '?'}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {error ? <div className="error">{error}</div> : null}
        {!selectedId ? (
          <div className="empty">Select an evaluation run, or start a new one.</div>
        ) : loadingRun || !run ? (
          <div className="empty">Loading run…</div>
        ) : (
          <>
            <header className="main-header">
              <div>
                <h2>{run.id}</h2>
                <p className="muted">
                  {formatTime(run.created_at)} · metric {run.metric ?? 'top5_recall'}
                  {run.parent_id ? ` · parent ${run.parent_id}` : ''}
                </p>
              </div>
              <button
                type="button"
                className="btn danger"
                disabled={deleting}
                onClick={() => void handleDeleteRun()}
              >
                {deleting ? 'Deleting…' : 'Delete run'}
              </button>
            </header>

            <div className="metrics">
              <div className="metric">
                <div className="metric-label">Score</div>
                <div className={`metric-value ${scoreClass(run.score)}`}>
                  {formatAggregateScore(run.score)}
                </div>
              </div>
              <div className="metric">
                <div className="metric-label">Examples</div>
                <div className="metric-value">{run.n_examples ?? results.length}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Perfect</div>
                <div className="metric-value">{run.n_perfect ?? '—'}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Zero</div>
                <div className="metric-value">{run.n_zero ?? '—'}</div>
              </div>
            </div>

            {run.notes ? <p className="notes">{run.notes}</p> : null}

            <details className="config">
              <summary>Config</summary>
              <pre className="block">{JSON.stringify(run.config ?? {}, null, 2)}</pre>
            </details>

            <section className="examples">
              <div className="examples-head">
                <h3>Examples</h3>
                <div className="filters">
                  {(['all', 'perfect', 'partial', 'failed'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      className={filter === mode ? 'chip active' : 'chip'}
                      onClick={() => setFilter(mode)}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {filtered.length === 0 ? (
                <p className="muted">No examples match this filter.</p>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Score</th>
                        <th>LLM</th>
                        <th>Claim</th>
                        <th>Gold</th>
                        <th>Pred (top-5)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map(({ row, index }) => (
                        <tr
                          key={index}
                          className={index === exampleIdx ? 'selected' : undefined}
                          onClick={() => setExampleIdx(index)}
                        >
                          <td>{index}</td>
                          <td className={scoreClass(row.score)}>{formatExampleScore(row.score)}</td>
                          <td>{row.n_llm_calls ?? row.llm_calls?.length ?? '—'}</td>
                          <td className="claim-cell">{(row.claim ?? '').slice(0, 100)}</td>
                          <td className="titles-cell">{(row.gold_titles ?? []).join(', ')}</td>
                          <td className="titles-cell">
                            {(row.pred_titles ?? []).slice(0, 5).join(', ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedExample ? (
                <ExampleDetail index={exampleIdx} row={selectedExample} />
              ) : null}
            </section>
          </>
        )}
      </main>
    </div>
  )
}
