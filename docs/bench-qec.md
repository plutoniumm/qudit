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

const qec = computed(() => data.value?.qec)

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({ name, values: v.mean_ms }))
}
</script>

# QEC Recovery Maps

Recovery map construction and application for qudit's three built-in methods across standard codes. Noise: amplitude damping at $\gamma = 0.05$.

> No cross-framework comparison — qudit is the only library with these methods natively.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code>.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

## Construction

<BenchChart
  v-if="qec?.construct"
  title="Recovery map construction — Petz / Leung / Cafaro"
  :series="series(qec.construct)"
  :xLabels="qec.construct.meta.configs"
  :note="`N=${qec?.construct?.meta?.N} runs. Includes Kraus stack + matrix ops (SVD, polar decomp).`"
/>

## Application

<BenchChart
  v-if="qec?.apply"
  title="Recovery channel application — Petz / Leung / Cafaro"
  :series="series(qec.apply)"
  :xLabels="qec.apply.meta.configs"
  :note="`N=${qec?.apply?.meta?.N} runs. Time per single rec.run(rho) call.`"
/>

**Notes**:
- **Petz** — pseudo-inverse recovery. Construction involves matrix eigendecomposition; application is a linear channel.
- **Leung** — polar decomposition recovery. SVD-based construction; similar application cost.
- **Cafaro** — normalisation recovery. Lightest construction; application equivalent to others.
- All three are $O(d^{2n})$ per application for $n$-qudit codes.
