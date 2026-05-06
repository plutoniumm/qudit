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

const gd = computed(() => data.value?.gd)

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({ name, values: v.mean_ms }))
}
</script>

# Gradient Descent (VQE)

Layered $R_Y$-$CX$ ansatz minimising $\langle Z \otimes I \otimes \cdots \rangle$. Full optimisation loop timed end-to-end.

`qudit` and PennyLane use PyTorch autograd — one backward pass per step. Others use parameter-shift — $2L$ forward passes per step where $L$ = number of parameters.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>cd bench && conda run -n qudit python run.py</code>.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

<BenchChart
  v-if="gd"
  title="VQE — full optimisation loop"
  :series="series(gd)"
  :xLabels="gd.meta.configs"
  :note="`N=${gd?.meta?.N} runs. AD = autograd (1 backward/step), PS = parameter-shift (2L fwd/step).`"
/>

**Notes**:
- `qudit (AD)` — `torch.autograd` through `nn.Module`. One backward pass per step regardless of parameter count.
- `pennylane (AD)` — same gradient method, higher constant overhead from the QNode abstraction.
- `qiskit (PS)` — `StatevectorEstimator` batch API with parameter-shift. Batches all shift evaluations per step.
- `cirq (PS)`, `braket (PS)`, `qutip (PS)` — parameter-shift with sequential circuit rebuilds per evaluation. Cost is $2L \times$ forward cost per step.
