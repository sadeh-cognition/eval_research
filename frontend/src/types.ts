export type RunSummary = {
  id: string
  created_at: string
  kind: string
  metric: string
  score: number | null
  n_examples: number
  n_perfect?: number | null
  n_zero?: number | null
  mean_example_score?: number | null
  parent_id?: string | null
  notes?: string | null
  student_lm?: string | null
  teacher_lm?: string | null
  backend?: string | null
  dev_size?: number | null
  train_size?: number | null
  auto?: string | null
  program_path?: string | null
  delta?: number | null
  baseline_score?: number | null
  path?: string | null
}

export type LlmMessage = {
  role: string
  content: string
}

export type LlmCall = {
  call_index: number
  uuid?: string
  timestamp?: string
  model?: string
  messages?: LlmMessage[]
  prompt?: string | null
  outputs?: string[]
  response_text?: string
  usage?: Record<string, unknown>
  cost?: number | null
}

export type ExampleResult = {
  claim?: string | null
  gold_titles?: string[]
  pred_titles?: string[]
  score?: number | null
  reasoning?: string | null
  trajectory?: Record<string, unknown> | unknown
  llm_calls?: LlmCall[]
  n_llm_calls?: number
  error?: string | null
}

export type EvalRun = {
  id: string
  created_at: string
  kind: string
  metric?: string
  score: number | null
  n_examples?: number
  n_perfect?: number | null
  n_zero?: number | null
  mean_example_score?: number | null
  parent_id?: string | null
  notes?: string | null
  config?: Record<string, unknown>
  results?: ExampleResult[]
  _path?: string
}

export type ModelOption = {
  id: string
  label: string
}

export type JobKind = 'eval' | 'optimize'

export type EvalJob = {
  id: string
  kind?: JobKind
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  created_at: string
  updated_at: string
  params: Record<string, unknown>
  error?: string | null
  run_id?: string | null
  baseline_run_id?: string | null
  run_ids?: string[]
  score?: number | null
  baseline_score?: number | null
  delta?: number | null
  path?: string | null
  program_path?: string | null
}

export type BackendName = 'auto' | 'colbert' | 'wikipedia'
export type AutoLevel = 'light' | 'medium' | 'heavy'

export type StartEvalPayload = {
  student_lm: string
  backend: BackendName
  train_size: number
  dev_size: number
  max_iters: number
  num_threads: number
  safe: boolean
  temperature: number
}

export type StartOptimizePayload = {
  student_lm: string
  teacher_lm: string
  backend: BackendName
  train_size: number
  dev_size: number
  max_iters: number
  num_threads: number
  safe: boolean
  temperature: number
  auto: AutoLevel
  max_bootstrapped_demos: number
  max_labeled_demos: number
  save?: string
}
