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

const noisy = computed(() => data.value?.noisy)

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({ name, values: v.mean_ms }))
}

function xq(sizes) { return sizes.map(s => s + 'q') }
</script>

# Noisy Circuits

GHZ circuit with noise applied per gate. All frameworks use their native noisy simulator.

> [!TIP]
> `qudit (stochastic)` is a **single shot** — one Kraus branch sampled per gate. `qudit (deterministic)` applies the full Kraus sum, directly comparable to qiskit/cirq/etc. Multiply stochastic time by shot count to get equivalent accuracy.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code>.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

---

## Depolarising

<BenchChart
  v-if="noisy?.depol"
  :title="noisy.depol.meta.label"
  :series="series(noisy.depol)"
  :xLabels="xq(noisy.depol.meta.sizes)"
  :note="`N=${noisy?.depol?.meta?.N} runs. qudit stochastic = single Weyl-channel shot.`"
/>

---

## Amplitude Damping

<BenchChart
  v-if="noisy?.ad"
  :title="noisy.ad.meta.label"
  :series="series(noisy.ad)"
  :xLabels="xq(noisy.ad.meta.sizes)"
  :note="`N=${noisy?.ad?.meta?.N} runs. qudit deterministic uses Process.AD full Kraus sum.`"
/>

---

## Phase Damping

<BenchChart
  v-if="noisy?.phase_damp"
  :title="noisy.phase_damp.meta.label"
  :series="series(noisy.phase_damp)"
  :xLabels="xq(noisy.phase_damp.meta.sizes)"
  :note="`N=${noisy?.phase_damp?.meta?.N} runs.`"
/>

---

**Notes**:
- `qudit (stochastic)` — `Mode.NOISY`, one Kraus branch sampled per gate. Constant time per shot, independent of noise strength. Scales to large $n$ because no full density matrix sum.
- `qudit (deterministic)` — full Kraus sum $\sum_k E_k \rho E_k^\dagger$ per channel application. Exact mixed state, directly comparable to qiskit/cirq. Cost scales as $O(K \cdot d^{2n})$.
- `qudit (mps)` — stochastic shot on Apple Silicon GPU. Overhead dominates for small circuits.
- All other frameworks compute the exact density matrix deterministically.
