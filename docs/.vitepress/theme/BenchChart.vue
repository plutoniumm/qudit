<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: String,
  // series: [{ name, values: number|null[], stds?: number[] }]
  series: Array,
  xLabels: Array,
  unit: { type: String, default: 'ms' },
  log: { type: Boolean, default: true },
  note: { type: String, default: '' },
})

const W = 720, H = 300
const PAD = { top: 20, right: 10, bottom: 44, left: 62 }
const CW = W - PAD.left - PAD.right
const CH = H - PAD.top - PAD.bottom

const COLORS = [
  '#7c3aed', '#e63946', '#2a9d8f', '#f4a261',
  '#457b9d', '#6a994e', '#e76f51', '#a8dadc',
]

const allVals = computed(() =>
  props.series.flatMap(s => s.values).filter(v => v != null && v > 0)
)

const yMin = computed(() => {
  const m = Math.min(...allVals.value)
  return props.log ? Math.pow(10, Math.floor(Math.log10(m))) : 0
})

const yMax = computed(() => {
  const m = Math.max(...allVals.value)
  return props.log ? Math.pow(10, Math.ceil(Math.log10(m * 1.2))) : m * 1.1
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

// y-axis tick values
const yTicks = computed(() => {
  if (!props.log) {
    const step = (yMax.value - yMin.value) / 5
    return Array.from({ length: 6 }, (_, i) => yMin.value + i * step)
  }
  const lo = Math.log10(yMin.value), hi = Math.log10(yMax.value)
  const ticks = []
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    ticks.push(Math.pow(10, e))
  }
  return ticks
})

function fmtY(v) {
  if (v >= 1000) return (v / 1000).toFixed(0) + 's'
  if (v >= 1) return v.toFixed(v < 10 ? 1 : 0)
  return v.toFixed(2)
}

function linePath(s) {
  const pts = s.values
    .map((v, i) => {
      const y = yScale(v)
      return y == null ? null : `${xScale(i).toFixed(1)},${y.toFixed(1)}`
    })
  const segments = []
  let current = []
  for (const pt of pts) {
    if (pt == null) {
      if (current.length > 1) segments.push('M ' + current.join(' L '))
      current = []
    } else {
      current.push(pt)
    }
  }
  if (current.length > 1) segments.push('M ' + current.join(' L '))
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
      <!-- grid -->
      <g class="grid">
        <line
          v-for="t in yTicks"
          :key="t"
          :x1="PAD.left" :x2="PAD.left + CW"
          :y1="yScale(t)" :y2="yScale(t)"
          stroke="currentColor" stroke-opacity="0.1" stroke-width="1"
        />
      </g>

      <!-- y-axis -->
      <g class="y-axis">
        <line :x1="PAD.left" :x2="PAD.left" :y1="PAD.top" :y2="PAD.top + CH"
          stroke="currentColor" stroke-opacity="0.25" stroke-width="1" />
        <g v-for="t in yTicks" :key="t">
          <line :x1="PAD.left - 4" :x2="PAD.left" :y1="yScale(t)" :y2="yScale(t)"
            stroke="currentColor" stroke-opacity="0.4" stroke-width="1" />
          <text :x="PAD.left - 7" :y="yScale(t)" dy="0.35em"
            text-anchor="end" font-size="11" fill="currentColor" opacity="0.6">
            {{ fmtY(t) }}
          </text>
        </g>
        <text :x="14" :y="PAD.top + CH / 2" text-anchor="middle"
          font-size="11" fill="currentColor" opacity="0.5"
          :transform="`rotate(-90, 14, ${PAD.top + CH / 2})`">
          {{ unit }}
        </text>
      </g>

      <!-- x-axis -->
      <g class="x-axis">
        <line :x1="PAD.left" :x2="PAD.left + CW" :y1="PAD.top + CH" :y2="PAD.top + CH"
          stroke="currentColor" stroke-opacity="0.25" stroke-width="1" />
        <g v-for="(label, i) in xLabels" :key="i">
          <line :x1="xScale(i)" :x2="xScale(i)"
            :y1="PAD.top + CH" :y2="PAD.top + CH + 4"
            stroke="currentColor" stroke-opacity="0.4" stroke-width="1" />
          <text :x="xScale(i)" :y="PAD.top + CH + 16"
            text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">
            {{ label }}
          </text>
        </g>
      </g>

      <!-- series lines + dots -->
      <g v-for="(s, si) in series" :key="s.name">
        <path
          :d="linePath(s)"
          :stroke="COLORS[si % COLORS.length]"
          stroke-width="2"
          fill="none"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
        <circle
          v-for="d in dots(s)"
          :key="`${d.cx}-${d.cy}`"
          :cx="d.cx" :cy="d.cy" r="3.5"
          :fill="COLORS[si % COLORS.length]"
        >
          <title>{{ s.name }}: {{ d.v.toFixed(2) }} {{ unit }}</title>
        </circle>
      </g>
    </svg>

    <!-- legend -->
    <div class="bench-legend">
      <span v-for="(s, si) in series" :key="s.name" class="legend-item">
        <svg width="20" height="4" style="vertical-align: middle; margin-right: 4px">
          <line x1="0" y1="2" x2="20" y2="2"
            :stroke="COLORS[si % COLORS.length]" stroke-width="2.5" />
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
  gap: 0.5rem 1.25rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  opacity: 0.8;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 2px;
}

.bench-note {
  font-size: 0.75rem;
  opacity: 0.55;
  margin-top: 0.5rem;
  font-style: italic;
}
</style>
