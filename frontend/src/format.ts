/** Aggregate dspy.Evaluate scores are percentages (e.g. 33.33); per-example are 0–1. */
export function formatAggregateScore(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return '—'
  if (score > 1) return `${score.toFixed(2)}%`
  return `${(score * 100).toFixed(2)}%`
}

export function formatExampleScore(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return '—'
  if (score > 1) return score.toFixed(2)
  return `${score.toFixed(3)} (${(score * 100).toFixed(1)}%)`
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  return iso.slice(0, 19).replace('T', ' ')
}

export function scoreClass(score: number | null | undefined): string {
  if (score == null) return 'score-muted'
  const s = score > 1 ? score / 100 : score
  if (s >= 0.99) return 'score-good'
  if (s >= 0.5) return 'score-mid'
  if (s > 0) return 'score-low'
  return 'score-zero'
}
