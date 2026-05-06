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

const qubo = computed(() => data.value?.qubo)

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({ name, values: v.mean_ms }))
}
</script>

# QUBO / QAOA

Benchmarks for QUBO-to-Ising conversion, QAOA forward pass, and ClockSolver optimisation. No cross-framework comparison — qudit is the only library with native qudit-general QAOA.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code>.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

## QUBO Conversion

Time to convert a random QUBO problem to an Ising Hamiltonian via `QUBO.toHamiltonian()`.

<BenchChart
  v-if="qubo?.convert"
  title="QUBO → Ising Hamiltonian conversion"
  :series="series(qubo.convert)"
  :xLabels="qubo.convert.meta.configs"
  :note="`N=${qubo?.convert?.meta?.N} runs. Random 50%-dense QUBO.`"
  :log="false"
/>

## QAOA Forward Pass

State construction time for the QAOA ansatz $|\psi(\gamma, \beta)\rangle$ per number of qubits and layers $p$.

<BenchChart
  v-if="qubo?.qaoa"
  title="QAOA forward pass"
  :series="series(qubo.qaoa)"
  :xLabels="qubo.qaoa.meta.configs"
  :note="`N=${qubo?.qaoa?.meta?.N} runs.`"
/>

## ClockSolver Optimisation

Full optimisation loop timing for the ClockSolver (QAOA variant with clock register) per qubit count and step count.

<BenchChart
  v-if="qubo?.clock"
  title="ClockSolver — full optimisation loop"
  :series="series(qubo.clock)"
  :xLabels="qubo.clock.meta.configs"
  :note="`N=${qubo?.clock?.meta?.N} runs.`"
/>
