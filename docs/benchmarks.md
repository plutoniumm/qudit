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

function series(section) {
  if (!section?.frameworks) return []
  return Object.entries(section.frameworks).map(([name, v]) => ({
    name,
    values: v.mean_ms,
  }))
}

function xq(sizes) {
  return sizes.map(s => s + 'q')
}

// Sections
const circuit   = computed(() => data.value?.circuit)
const noisy     = computed(() => data.value?.noisy)
const gd        = computed(() => data.value?.gd)
const qec       = computed(() => data.value?.qec)
</script>

# Benchmarks

All results are wall-clock means over ≥20 runs with warmup, on the same hardware.
The JSON is produced by `cd bench && conda run -n qudit python run.py` and served from `docs/public/bench.json`.

<div v-if="error" class="custom-block danger">
  <p><strong>bench.json not found:</strong> {{ error }}<br>Run <code>./do bench</code> to generate it.</p>
</div>
<div v-else-if="!data" style="opacity:0.5;font-style:italic;margin:2rem 0">Loading…</div>

---

## Ideal circuit — statevector

GHZ circuit ($H_0, CX_{01}, \ldots, CX_{n-2,n-1}$) in statevector mode. All frameworks use their default CPU statevector simulator. `qudit (mps)` runs on Apple Silicon GPU via Metal.

<BenchChart
  v-if="circuit"
  title="Ideal GHZ circuit — CPU vs MPS vs other frameworks"
  :series="series(circuit)"
  :xLabels="xq(circuit.meta.sizes)"
  :note="`N=${circuit?.meta?.N} runs, warmup=${circuit?.meta?.warmup}. Log scale. Lower is better.`"
/>

**Caveats**:
- `qudit (cpu)` applies gates via index-aware tensor contraction; constant overhead is higher but Hilbert-space scaling is better than dense Kronecker products.
- `qudit (mps)` transfers tensors to Metal GPU — wins above ~8 qubits where data movement is amortised.
- `numpy` builds the full embedded unitary per gate (dense Kronecker), fast at small $n$, bad scaling.
- `braket` and `qiskit` include IR serialisation overhead.
- PennyLane's `lightning.qubit` is JIT-compiled; typically fastest past 8 qubits on CPU.

---

## Noisy circuits

Each framework uses its **native** implementation of the corresponding channel applied per gate on a GHZ circuit. `qudit` uses `Mode.NOISY` with stochastic Kraus sampling — one trajectory per forward pass, not the full deterministic Kraus sum.

> [!TIP]
> `qudit`'s `Mode.NOISY` is a **single stochastic shot**. To compare fairly with deterministic simulators, multiply qudit's time by the number of shots needed.

### Depolarising

<BenchChart
  v-if="noisy?.depol"
  :title="noisy.depol.meta.label"
  :series="series(noisy.depol)"
  :xLabels="xq(noisy.depol.meta.sizes)"
  :note="`N=${noisy?.depol?.meta?.N} runs. qudit: single stochastic shot (WeylNoise).`"
/>

### Amplitude Damping

<BenchChart
  v-if="noisy?.ad"
  :title="noisy.ad.meta.label"
  :series="series(noisy.ad)"
  :xLabels="xq(noisy.ad.meta.sizes)"
  :note="`N=${noisy?.ad?.meta?.N} runs. qudit: PhysicalNoise with T1/T2 derived from γ.`"
/>

### Phase Damping

<BenchChart
  v-if="noisy?.phase_damp"
  :title="noisy.phase_damp.meta.label"
  :series="series(noisy.phase_damp)"
  :xLabels="xq(noisy.phase_damp.meta.sizes)"
  :note="`N=${noisy?.phase_damp?.meta?.N} runs.`"
/>

---

## QEC recovery maps

Recovery map construction and application time for `qudit`'s three built-in methods on standard codes. Noise: amplitude-damping at $\gamma = 0.05$.

> No cross-framework comparison — `qudit` is the only library with these methods natively. These numbers show the relative cost of Petz (pseudo-inverse), Leung (polar decomp), and Cafaro (normalization).

### Construction time

<BenchChart
  v-if="qec?.construct"
  title="Recovery map construction — Petz / Leung / Cafaro"
  :series="series(qec.construct)"
  :xLabels="qec.construct.meta.configs"
  :note="`N=${qec?.construct?.meta?.N} runs. Includes Kraus stack + matrix ops.`"
/>

### Application time (per `rec.run(rho)`)

<BenchChart
  v-if="qec?.apply"
  title="Recovery channel application — Petz / Leung / Cafaro"
  :series="series(qec.apply)"
  :xLabels="qec.apply.meta.configs"
  :note="`N=${qec?.apply?.meta?.N} runs. Time per single rec.run(rho) call.`"
/>

---

## Gradient descent (VQE)

Layered $R_Y$-$CX$ ansatz, minimising $\langle Z \otimes I \otimes \cdots \rangle$. Full optimisation loop timed per run.

`qudit` and PennyLane use PyTorch autograd (one backward pass per step). Others use parameter-shift (2 evals/parameter/step).

<BenchChart
  v-if="gd"
  title="VQE — full optimisation loop"
  :series="series(gd)"
  :xLabels="gd.meta.configs"
  :note="`N=${gd?.meta?.N} runs. AD = autograd, PS = parameter-shift (2× fwd cost per param/step).`"
/>

**Caveats**:
- `qudit` uses `torch.autograd` through `nn.Module` — one backward pass per step.
- Qiskit uses `StatevectorEstimator` batch API with parameter-shift.
- Braket and cirq rebuild the circuit per evaluation in Python.
