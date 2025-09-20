import { ref } from 'vue'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8001'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
})

export function useLLMApi() {
  const isLoading = ref(false)
  const error = ref(null)

  const generateText = async (params, onToken = null) => {
    isLoading.value = true
    error.value = null

    try {
      if (params.stream) {
        return await generateStreaming(params, onToken)
      } else {
        return await generateNonStreaming(params)
      }
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const generateStreaming = async (params, onToken) => {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: params.prompt,
        max_tokens: params.max_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        stream: true
      })
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let result = {
      text: '',
      tokens_generated: 0,
      generation_time: 0,
      cached: false
    }
    
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      
      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            
            if (data.token && onToken) {
              onToken(data.token)
              result.text += data.token
            }
            
            if (data.token_count) {
              result.tokens_generated = data.token_count
            }
            
            if (data.generation_time) {
              result.generation_time = data.generation_time
            }
            
            if (data.cached !== undefined) {
              result.cached = data.cached
            }
          } catch (e) {
            // Ignore parsing errors for non-JSON lines
            console.warn('Failed to parse SSE data:', e)
          }
        }
      }
    }
    
    return result
  }

  const generateNonStreaming = async (params) => {
    const response = await api.post('/generate', {
      prompt: params.prompt,
      max_tokens: params.max_tokens,
      temperature: params.temperature,
      top_p: params.top_p,
      top_k: params.top_k,
      stream: false
    })
    
    return response.data
  }

  const checkHealth = async () => {
    try {
      const response = await api.get('/health')
      return response.data
    } catch (err) {
      throw new Error(`Health check failed: ${err.message}`)
    }
  }

  const getModelInfo = async () => {
    try {
      const response = await api.get('/models/info')
      return response.data
    } catch (err) {
      throw new Error(`Model info failed: ${err.message}`)
    }
  }

  return {
    isLoading,
    error,
    generateText,
    checkHealth,
    getModelInfo
  }
}
