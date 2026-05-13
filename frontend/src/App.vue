<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { api, type PredictResponse, type HealthResponse } from './lib/api'
import { loadHistory, pushHistory, clearHistory, type HistoryItem } from './lib/storage'

const text = ref('')
const threshold = ref(0.5)
const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<PredictResponse | null>(null)
const health = ref<HealthResponse | null>(null)
const history = ref<HistoryItem[]>([])

const sortedLabels = computed(() => {
  if (!result.value) return []
  return [...result.value.labels].sort((a, b) => b.probability - a.probability)
})

async function submit() {
  if (!text.value.trim()) return
  loading.value = true
  error.value = null
  try {
    const r = await api.predict(text.value, threshold.value)
    result.value = r
    history.value = pushHistory(r)
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

function reuseHistory(item: HistoryItem) {
  text.value = item.result.text
  threshold.value = item.result.threshold
  result.value = item.result
}

function onClearHistory() {
  clearHistory()
  history.value = []
}

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}

onMounted(async () => {
  history.value = loadHistory()
  try {
    health.value = await api.health()
  } catch (e) {
    error.value = `API 연결 실패: ${(e as Error).message}`
  }
})
</script>

<template>
  <div class="min-h-screen p-6 max-w-3xl mx-auto">
    <header class="mb-6">
      <h1 class="text-2xl font-bold text-slate-800">한국어 혐오 표현 분류기</h1>
      <p class="text-sm text-slate-500 mt-1">
        KoELECTRA 기반 10개 카테고리 다중 레이블 분류
        <span v-if="health" class="ml-2 inline-block rounded bg-emerald-100 text-emerald-800 px-2 py-0.5 text-xs">
          {{ health.device }} · {{ health.model_loaded ? '준비됨' : '로딩 중' }}
        </span>
      </p>
    </header>

    <section class="bg-white rounded-lg shadow-sm border border-slate-200 p-4 mb-4">
      <label class="block text-sm font-medium text-slate-700 mb-2">분류할 텍스트</label>
      <textarea
        v-model="text"
        rows="4"
        placeholder="분류할 텍스트를 입력하세요"
        class="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
        @keydown.ctrl.enter="submit"
        @keydown.meta.enter="submit"
      />

      <div class="mt-3 flex items-center gap-4">
        <label class="text-sm text-slate-600 flex items-center gap-2 flex-1">
          임계값
          <input v-model.number="threshold" type="range" min="0" max="1" step="0.05" class="flex-1" />
          <span class="font-mono text-xs w-12 text-right">{{ threshold.toFixed(2) }}</span>
        </label>
        <button
          :disabled="loading || !text.trim()"
          class="rounded bg-emerald-600 text-white text-sm px-4 py-2 disabled:opacity-50 hover:bg-emerald-700"
          @click="submit"
        >
          {{ loading ? '분류 중…' : '분류' }}
        </button>
      </div>
      <p v-if="error" class="mt-3 text-sm text-rose-600">{{ error }}</p>
    </section>

    <section v-if="result" class="bg-white rounded-lg shadow-sm border border-slate-200 p-4 mb-4">
      <div class="mb-3 flex items-center justify-between">
        <h2 class="font-semibold text-slate-800">분류 결과</h2>
        <span class="text-sm text-slate-500">
          최상위: <b class="text-slate-800">{{ result.top.name }}</b>
          ({{ pct(result.top.probability) }})
        </span>
      </div>
      <ul class="space-y-1.5">
        <li v-for="l in sortedLabels" :key="l.name" class="text-sm">
          <div class="flex items-center justify-between">
            <span :class="l.predicted ? 'font-semibold text-slate-800' : 'text-slate-600'">
              {{ l.name }}
            </span>
            <span class="font-mono text-xs text-slate-500">{{ pct(l.probability) }}</span>
          </div>
          <div class="h-2 bg-slate-100 rounded overflow-hidden">
            <div
              class="h-full"
              :class="l.predicted ? 'bg-rose-500' : 'bg-slate-300'"
              :style="{ width: `${(l.probability * 100).toFixed(1)}%` }"
            />
          </div>
        </li>
      </ul>
    </section>

    <section v-if="history.length" class="bg-white rounded-lg shadow-sm border border-slate-200 p-4">
      <div class="mb-2 flex items-center justify-between">
        <h2 class="font-semibold text-slate-800 text-sm">최근 분류</h2>
        <button class="text-xs text-slate-500 hover:text-rose-600" @click="onClearHistory">
          전체 삭제
        </button>
      </div>
      <ul class="divide-y divide-slate-100">
        <li
          v-for="item in history"
          :key="item.at"
          class="py-2 cursor-pointer hover:bg-slate-50 rounded px-2"
          @click="reuseHistory(item)"
        >
          <div class="flex items-center justify-between gap-2">
            <span class="truncate text-sm text-slate-700">{{ item.result.text }}</span>
            <span class="text-xs text-slate-500 whitespace-nowrap">
              {{ item.result.top.name }} ({{ pct(item.result.top.probability) }})
            </span>
          </div>
        </li>
      </ul>
    </section>
  </div>
</template>
