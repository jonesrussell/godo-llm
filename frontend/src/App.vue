<template>
  <div class="container">
    <div class="header">
      <h1>🤖 Local LLM Inference</h1>
      <p>Test your local Llama-2-7b model with real-time streaming</p>
    </div>

    <div class="input-section">
      <label for="prompt">Prompt:</label>
      <textarea 
        id="prompt"
        v-model="prompt" 
        placeholder="Enter your prompt here... (e.g., 'Explain quantum computing in simple terms')"
        :disabled="isGenerating"
      ></textarea>
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Max Tokens</label>
        <input type="number" v-model.number="maxTokens" min="1" max="2048" :disabled="isGenerating">
      </div>
      <div class="control-group">
        <label>Temperature</label>
        <input type="number" v-model.number="temperature" min="0" max="2" step="0.1" :disabled="isGenerating">
      </div>
      <div class="control-group">
        <label>Top P</label>
        <input type="number" v-model.number="topP" min="0" max="1" step="0.05" :disabled="isGenerating">
      </div>
      <div class="control-group">
        <label>Top K</label>
        <input type="number" v-model.number="topK" min="1" max="100" :disabled="isGenerating">
      </div>
      <div class="control-group">
        <label>Stream</label>
        <select v-model="stream" :disabled="isGenerating">
          <option :value="true">Yes</option>
          <option :value="false">No</option>
        </select>
      </div>
    </div>

    <div class="button-group">
      <button 
        class="btn btn-primary" 
        @click="generateText" 
        :disabled="isGenerating || !prompt.trim()"
      >
        <span v-if="isGenerating" class="loading"></span>
        {{ isGenerating ? 'Generating...' : 'Generate' }}
      </button>
      <button 
        class="btn btn-secondary" 
        @click="clearOutput" 
        :disabled="isGenerating"
      >
        Clear
      </button>
      <button 
        class="btn btn-secondary" 
        @click="checkHealth" 
        :disabled="isGenerating"
      >
        Health Check
      </button>
    </div>

    <div class="output-section">
      <label>Output:</label>
      <div class="output">{{ output }}</div>
      
      <div v-if="status" :class="['status', status.type]">
        <span v-if="status.type === 'success'">✅</span>
        <span v-else-if="status.type === 'error'">❌</span>
        <span v-else>ℹ️</span>
        {{ status.message }}
      </div>
      
      <div v-if="stats" class="stats">
        <div class="stat">
          <span class="stat-value">{{ stats.tokens_generated || 0 }}</span>
          <span class="stat-label">Tokens</span>
        </div>
        <div class="stat">
          <span class="stat-value">{{ stats.generation_time ? stats.generation_time.toFixed(2) : '0.00' }}</span>
          <span class="stat-label">Seconds</span>
        </div>
        <div class="stat">
          <span class="stat-value">{{ stats.tokens_per_second ? stats.tokens_per_second.toFixed(1) : '0.0' }}</span>
          <span class="stat-label">Tokens/s</span>
        </div>
        <div class="stat" v-if="stats.cached">
          <span class="stat-value">🚀</span>
          <span class="stat-label">Cached</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useLLMApi } from './composables/useLLMApi'

export default {
  name: 'App',
  setup() {
    const prompt = ref('Explain quantum computing in simple terms')
    const maxTokens = ref(256)
    const temperature = ref(0.7)
    const topP = ref(0.9)
    const topK = ref(40)
    const stream = ref(true)
    const output = ref('')
    const isGenerating = ref(false)
    const status = ref(null)
    const stats = ref(null)

    const { generateText: apiGenerateText, checkHealth: apiCheckHealth } = useLLMApi()

    const generateText = async () => {
      if (!prompt.value.trim()) return
      
      isGenerating.value = true
      output.value = ''
      status.value = null
      stats.value = null
      
      try {
        const result = await apiGenerateText({
          prompt: prompt.value,
          max_tokens: maxTokens.value,
          temperature: temperature.value,
          top_p: topP.value,
          top_k: topK.value,
          stream: stream.value
        }, (token) => {
          if (stream.value) {
            output.value += token
          }
        })

        if (!stream.value) {
          output.value = result.text
        }
        
        stats.value = {
          tokens_generated: result.tokens_generated,
          generation_time: result.generation_time,
          tokens_per_second: result.tokens_generated / result.generation_time,
          cached: result.cached
        }
        
        status.value = {
          type: 'success',
          message: 'Generation completed successfully!'
        }
      } catch (error) {
        status.value = {
          type: 'error',
          message: `Error: ${error.message}`
        }
      } finally {
        isGenerating.value = false
      }
    }
    
    const clearOutput = () => {
      output.value = ''
      status.value = null
      stats.value = null
    }
    
    const checkHealth = async () => {
      try {
        const health = await apiCheckHealth()
        status.value = {
          type: 'info',
          message: `API Status: ${health.status}, Model Loaded: ${health.model_loaded}, Redis: ${health.redis_connected}`
        }
      } catch (error) {
        status.value = {
          type: 'error',
          message: `Health check failed: ${error.message}`
        }
      }
    }

    onMounted(() => {
      checkHealth()
    })

    return {
      prompt,
      maxTokens,
      temperature,
      topP,
      topK,
      stream,
      output,
      isGenerating,
      status,
      stats,
      generateText,
      clearOutput,
      checkHealth
    }
  }
}
</script>
