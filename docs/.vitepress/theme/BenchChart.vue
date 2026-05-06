<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: String,
  series: Array,       // [{ name, values: (number|null)[], stds?: number[] }]
  xLabels: Array,
  unit: { type: String, default: 'ms' },
  log: { type: Boolean, default: true },
  note: { type: String, default: '' },
})

const W = 720, H = 320
const PAD = { top: 24, right: 16, bottom: 48, left: 66 }
const CW = W - PAD.left - PAD.right
const CH = H - PAD.top - PAD.bottom

// qudit series get special treatment
function isOurs(name) {
  return name != null && name.toString().toLowerCase().startsWith('qudit')
}

const COLORS_OURS  = ['#7c3aed', '#a855f7']   // purple family for qudit variants
const COLORS_OTHER = ['#94a3b8', '#64748b', '#475569', '#93c5fd', '#6ee7b7', '#fca5a5', '#fcd34d', '#a5b4fc']

function seriesColor(name, idx) {
  const ourIdx = props.series.filter(s => isOurs(s.name)).indexOf(props.series.find(s => s.name === name))
  if (isOurs(name)) return COLORS_OURS[ourIdx % COLORS_OURS.length]
  const otherIdx = props.series.filter(s => !isOurs(s.name)).indexOf(props.series.find(s => s.name === name))
  return COLORS_OTHER[otherIdx % COLORS_OTHER.length]
}

const allVals = computed(() =>
  props.series.flatMap(s => s.values).filter(v => v != null && v > 0)
)

const yMin = computed(() => {
  const m = Math.min(...allVals.value)
  return props.log ? Math.pow(10, Math.floor(Math.log10(m))) : 0
})

const yMax = computed(() => {
  const m = Math.max(...allVals.value)
  return props.log ? Math.pow(10, Math.ceil(Math.log10(m * 1.5))) : m * 1.1
})

function yScale(v) {
  if (v == null || v <= 0) return null
  if (props.log) {
    const lo = Math.log10(yMin.value), hi = Math.log10(yMax.value)
    return PAD.top + CH * (1 - (Math.log10(v) - lo) / (hi - lo))
  }
  return PAD.top + CH * (1 - (v - yMin.value) / (yMax.value - yMin.value))
}

function xScale(i) {
  const n = props.xLabels.length
  return PAD.left + (n === 1 ? CW / 2 : i * CW / (n - 1))
}

const yTicks = computed(() => {
  if (!props.log) {
    const step = (yMax.value - yMin.value) / 5
    return Array.from({ length: 6 }, (_, i) => yMin.value + i * step)
  }
  const lo = Math.log10(yMin.value), hi = Math.log10(yMax.value)
  const ticks = []
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) ticks.push(Math.pow(10, e))
  return ticks
})

function fmtY(v) {
  if (v >= 1000) return (v / 1000).toFixed(0) + 's'
  if (v >= 1) return v.toFixed(v < 10 ? 1 : 0)
  return v.toFixed(2)
}

function linePath(s) {
  const pts = s.values.map((v, i) => {
    const y = yScale(v)
    return y == null ? null : `${xScale(i).toFixed(1)},${y.toFixed(1)}`
  })
  const segments = []
  let cur = []
  for (const pt of pts) {
    if (pt == null) { if (cur.length > 1) segments.push('M ' + cur.join(' L ')); cur = [] }
    else cur.push(pt)
  }
  if (cur.length > 1) segments.push('M ' + cur.join(' L '))
  return segments.join(' ')
}

function dots(s) {
  return s.values.map((v, i) => {
    const y = yScale(v)
    return y == null ? null : { cx: xScale(i), cy: y, v }
  }).filter(Boolean)
}
</script>

<template>
  <figure class="bench-chart">
    <figcaption v-if="title">{{ title }}</figcaption>
    <svg :viewBox="`0 0 ${W} ${H}`" :width="W" :height="H" class="bench-svg">

      <!-- background band for qudit series area -->
      <rect
        v-if="series.some(s => isOurs(s.name))"
        :x="PAD.left" :y="PAD.top" :width="CW" :height="CH"
        fill="#7c3aed" fill-opacity="0.03" rx="2"
      />

      <!-- grid -->
      <g class="grid">
        <line
          v-for="t in yTicks" :key="t"
          :x1="PAD.left" :x2="PAD.left + CW"
          :y1="yScale(t)" :y2="yScale(t)"
          stroke="currentColor" stroke-opacity="0.08" stroke-width="1"
        />
      </g>

      <!-- y-axis -->
      <g class="y-axis">
        <line :x1="PAD.left" :x2="PAD.left" :y1="PAD.top" :y2="PAD.top + CH"
          stroke="currentColor" stroke-opacity="0.2" stroke-width="1" />
        <g v-for="t in yTicks" :key="t">
          <line :x1="PAD.left - 4" :x2="PAD.left" :y1="yScale(t)" :y2="yScale(t)"
            stroke="currentColor" stroke-opacity="0.3" stroke-width="1" />
          <text :x="PAD.left - 8" :y="yScale(t)" dy="0.35em"
            text-anchor="end" font-size="11" fill="currentColor" opacity="0.55">
            {{ fmtY(t) }}
          </text>
        </g>
        <text :x="14" :y="PAD.top + CH / 2" text-anchor="middle"
          font-size="11" fill="currentColor" opacity="0.45"
          :transform="`rotate(-90, 14, ${PAD.top + CH / 2})`">
          {{ unit }}
        </text>
      </g>

      <!-- x-axis -->
      <g class="x-axis">
        <line :x1="PAD.left" :x2="PAD.left + CW" :y1="PAD.top + CH" :y2="PAD.top + CH"
          stroke="currentColor" stroke-opacity="0.2" stroke-width="1" />
        <g v-for="(label, i) in xLabels" :key="i">
          <line :x1="xScale(i)" :x2="xScale(i)"
            :y1="PAD.top + CH" :y2="PAD.top + CH + 4"
            stroke="currentColor" stroke-opacity="0.3" stroke-width="1" />
          <text :x="xScale(i)" :y="PAD.top + CH + 16"
            text-anchor="middle" font-size="11" fill="currentColor" opacity="0.55">
            {{ label }}
          </text>
        </g>
      </g>

      <!-- other frameworks first (drawn below qudit) -->
      <g v-for="(s, si) in series.filter(s => !isOurs(s.name))" :key="'o'+s.name">
        <path
          :d="linePath(s)"
          :stroke="seriesColor(s.name, si)"
          stroke-width="1.5"
          stroke-dasharray="5,3"
          fill="none"
          stroke-linejoin="round" stroke-linecap="round"
          opacity="0.7"
        />
        <circle
          v-for="d in dots(s)" :key="`${d.cx}-${d.cy}`"
          :cx="d.cx" :cy="d.cy" r="2.5"
          :fill="seriesColor(s.name, si)" opacity="0.7"
        >
          <title>{{ s.name }}: {{ d.v.toFixed(3) }} {{ unit }}</title>
        </circle>
      </g>

      <!-- qudit series on top — solid, thicker, prominent -->
      <g v-for="(s, si) in series.filter(s => isOurs(s.name))" :key="'q'+s.name">
        <path
          :d="linePath(s)"
          :stroke="seriesColor(s.name, si)"
          stroke-width="3"
          fill="none"
          stroke-linejoin="round" stroke-linecap="round"
        />
        <circle
          v-for="d in dots(s)" :key="`${d.cx}-${d.cy}`"
          :cx="d.cx" :cy="d.cy" r="4.5"
          :fill="seriesColor(s.name, si)"
          stroke="white" stroke-width="1.5"
        >
          <title>{{ s.name }}: {{ d.v.toFixed(3) }} {{ unit }}</title>
        </circle>
      </g>

    </svg>

    <!-- legend: qudit first, then others -->
    <div class="bench-legend">
      <span
        v-for="s in [...series.filter(s => isOurs(s.name)), ...series.filter(s => !isOurs(s.name))]"
        :key="s.name"
        class="legend-item"
        :class="{ 'legend-ours': isOurs(s.name) }"
      >
        <svg width="22" height="6" style="vertical-align:middle;margin-right:4px">
          <line x1="0" y1="3" x2="22" y2="3"
            :stroke="seriesColor(s.name, 0)"
            :stroke-width="isOurs(s.name) ? 3 : 1.5"
            :stroke-dasharray="isOurs(s.name) ? 'none' : '5,3'"
          />
        </svg>
        {{ s.name }}
      </span>
    </div>

    <p v-if="note" class="bench-note">{{ note }}</p>
  </figure>
</template>

<style scoped>
.bench-chart {
  margin: 1.5rem 0;
  padding: 0;
}

figcaption {
  font-size: 0.85rem;
  font-weight: 600;
  opacity: 0.75;
  margin-bottom: 0.5rem;
}

.bench-svg {
  display: block;
  width: 100%;
  height: auto;
  max-width: 720px;
}

.bench-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.1rem;
  margin-top: 0.6rem;
  font-size: 0.8rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 2px;
  opacity: 0.7;
}

.legend-ours {
  opacity: 1;
  font-weight: 600;
  color: #7c3aed;
}

.bench-note {
  font-size: 0.75rem;
  opacity: 0.5;
  margin-top: 0.4rem;
  font-style: italic;
}
</style>
