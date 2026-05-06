<script setup>
import { ref, computed, onMounted } from 'vue'
import BenchChart from './.vitepress/theme/BenchChart.vue'

const data = ref(null)
const error = ref(null)

onMounted(async () => {
  try {
    const base = import.meta.env.BASE_URL || '/'
    const res = await fetch(`${base}bench.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    data.value = await res.json()
  } catch (e) {
    error.value = e.message
  }
})

const metrics = computed(() => data.value?.metrics)

function series(label) {
  if (!metrics.value?.results?.[label]) return []
  const r = metrics.value.results[label]
  return [{ name: 'qudit', values: r.mean_ms }]
}

const xLabels = computed(() =>
  metrics.value?.meta?.sizes?.map(d => `d=${d}`) ?? []
)
</script>

# Metrics

Timing for qudit's built-in quantum information metrics on random density matrices of increasing dimension $d \times d$.

No cross-framework comparison — benchmarks show internal scaling behaviour.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code>.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

## Fidelity

Uhlmann fidelity $F(\rho, \sigma) = \left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2$.

<BenchChart
  v-if="metrics"
  title="Uhlmann Fidelity"
  :series="series('fidelity')"
  :xLabels="xLabels"
  :note="`N=${metrics?.meta?.N} runs. Two random density matrices per call.`"
/>

## Von Neumann Entropy

$S(\rho) = -\mathrm{Tr}(\rho \log \rho)$.

<BenchChart
  v-if="metrics"
  title="Von Neumann Entropy"
  :series="series('entropy')"
  :xLabels="xLabels"
  :note="`N=${metrics?.meta?.N} runs.`"
/>

## Mutual Information

$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$ on a $d^2 \times d^2$ bipartite system.

<BenchChart
  v-if="metrics"
  title="Mutual Information"
  :series="series('mutual_info')"
  :xLabels="xLabels"
  :note="`N=${metrics?.meta?.N} runs. Matrix is d²×d² (bipartite).`"
/>

## Trace Distance

$D(\rho, \sigma) = \frac{1}{2}\mathrm{Tr}|\rho - \sigma|$.

<BenchChart
  v-if="metrics"
  title="Trace Distance"
  :series="series('trace_dist')"
  :xLabels="xLabels"
  :note="`N=${metrics?.meta?.N} runs.`"
/>

## Bures Distance

$D_B(\rho,\sigma) = \sqrt{2 - 2\sqrt{F(\rho,\sigma)}}$.

<BenchChart
  v-if="metrics"
  title="Bures Distance"
  :series="series('bures')"
  :xLabels="xLabels"
  :note="`N=${metrics?.meta?.N} runs.`"
/>
