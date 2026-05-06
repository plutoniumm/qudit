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

const circuit = computed(() => data.value?.circuit)

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({ name, values: v.mean_ms }))
}

function xq(sizes) { return sizes.map(s => s + 'q') }
</script>

# Ideal Circuit

GHZ circuit ($H_0, CX_{01}, \ldots, CX_{n-2,n-1}$) in statevector mode across n = 2 … 15 qubits.

All frameworks use their default CPU statevector simulator. `qudit (mps)` offloads to Apple Silicon GPU via Metal.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code> to generate it.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

<BenchChart
  v-if="circuit"
  title="GHZ circuit — statevector simulation"
  :series="series(circuit)"
  :xLabels="xq(circuit.meta.sizes)"
  :note="`N=${circuit?.meta?.N} runs, warmup=${circuit?.meta?.warmup}. Log scale. Lower is better.`"
/>

**Notes**:
- `qudit (cpu)` uses index-aware tensor contraction — constant overhead is higher than dense Kronecker at small $n$, but Hilbert-space scaling is better past $n \approx 8$.
- `qudit (mps)` transfers tensors to Metal GPU — overhead dominates at small $n$, wins when statevector exceeds L3 cache (~$n \geq 12$).
- `numpy` builds the full embedded unitary per gate (dense Kronecker); fast at small $n$, exponential blow-up after $n = 8$.
- PennyLane `lightning.qubit` is JIT-compiled C++; typically fastest on CPU past 8 qubits among third-party frameworks.
- `braket` and `qiskit` include IR serialisation overhead in each run.
