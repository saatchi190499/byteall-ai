import { useMemo, useState } from 'react'

const RAW_API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.port === '5173' ? 'http://127.0.0.1:8000' : '')
const API_BASE = normalizeHttpBase(RAW_API_BASE)

export default function App() {
  const [messages, setMessages] = useState([
    { id: 'boot', role: 'assistant', content: 'Index your PDFs, then ask me to generate code.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [code, setCode] = useState('')
  const [useRag, setUseRag] = useState(true)

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading])

  const runIndex = async () => {
    setIndexing(true)
    try {
      const startRes = await safeFetch('/api/index', { method: 'POST' })
      const startData = await startRes.json()
      if (!startRes.ok) throw new Error(startData.detail || 'Failed to start indexing')

      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'assistant',
          content: `Index job started: ${startData.job_id}`,
        },
      ])

      const data = await waitForIndexJob(startData.job_id)
      const skippedCount = Array.isArray(data.skipped_files) ? data.skipped_files.length : 0
      const skippedInfo =
        skippedCount > 0 ? ` Skipped ${skippedCount} unreadable file(s): ${data.skipped_files.join(', ')}` : ''

      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'assistant',
          content: `Indexed ${data.indexed_files} file(s), ${data.indexed_chunks} chunk(s).${skippedInfo}`,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'assistant',
          content: `Index error: ${err.message}`,
        },
      ])
    } finally {
      setIndexing(false)
    }
  }

  const sendMessage = async () => {
    if (!canSend) return
    const userMessage = input.trim()
    setInput('')
    setLoading(true)
    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: 'user', content: userMessage },
    ])

    try {
      let data
      try {
        data = await sendViaWebSocket({ message: userMessage, use_rag: useRag, editor_code: code })
      } catch (_wsErr) {
        const res = await safeFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage, use_rag: useRag, editor_code: code }),
        })
        data = await res.json()
        if (!res.ok) throw new Error(data.detail || 'Failed to chat')
      }

      if (data.code) {
        setCode(data.code)
      }
      setMessages((prev) => [...prev, { id: `asst-${Date.now()}`, role: 'assistant', content: data.answer }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'assistant', content: `Chat error: ${err.message}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <h1>AI Code Agent</h1>
        <button onClick={runIndex} disabled={indexing}>
          {indexing ? 'Indexing...' : 'Index PDFs'}
        </button>
      </header>

      <main className="split-layout">
        <section className="panel code-panel">
          <h2>Notebook Editor</h2>
          <textarea
            className="code-editor"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Write or paste your code here. The AI will read this notebook editor content."
            spellCheck={false}
          />
        </section>

        <section className="panel chat-panel">
          <h2>Chat</h2>
          <div className="messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <strong>{msg.role === 'user' ? 'You' : 'Agent'}</strong>
                <p>{msg.content}</p>
              </div>
            ))}
          </div>

          <form onSubmit={onSubmit} className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                useRag
                  ? 'Ask for code generation with context from your PDFs...'
                  : 'Ask for code generation without PDF context...'
              }
              rows={3}
            />
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
              />
              Use RAG (PDF context)
            </label>
            <button type="submit" disabled={!canSend}>
              {loading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </section>
      </main>
    </div>
  )
}

function getWebSocketUrl() {
  if (API_BASE) {
    const base = API_BASE.replace(/^http/, 'ws')
    return `${base}/api/ws/chat`
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/api/ws/chat`
}

function apiUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path
}

async function safeFetch(path, options) {
  const primary = apiUrl(path)
  try {
    return await fetch(primary, options)
  } catch (primaryErr) {
    if (primary === path) {
      throw primaryErr
    }
    return fetch(path, options)
  }
}

function normalizeHttpBase(base) {
  const trimmed = String(base || '').trim()
  if (!trimmed) return ''
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '')
  if (trimmed.startsWith('//')) return `${window.location.protocol}${trimmed}`.replace(/\/+$/, '')
  return `http://${trimmed}`.replace(/\/+$/, '')
}

function sendViaWebSocket(payload) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(getWebSocketUrl())
    let settled = false

    const timeout = setTimeout(() => {
      if (!settled) {
        settled = true
        ws.close()
        reject(new Error('WebSocket timeout'))
      }
    }, 120000)

    ws.onopen = () => {
      ws.send(JSON.stringify(payload))
    }

    ws.onmessage = (event) => {
      if (settled) return
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
          settled = true
          clearTimeout(timeout)
          ws.close()
          reject(new Error(data.error))
          return
        }
        settled = true
        clearTimeout(timeout)
        ws.close()
        resolve(data)
      } catch (err) {
        settled = true
        clearTimeout(timeout)
        ws.close()
        reject(err)
      }
    }

    ws.onerror = () => {
      if (!settled) {
        settled = true
        clearTimeout(timeout)
        reject(new Error('WebSocket connection failed'))
      }
    }

    ws.onclose = () => {
      if (!settled) {
        settled = true
        clearTimeout(timeout)
        reject(new Error('WebSocket closed before response'))
      }
    }
  })
}

async function waitForIndexJob(jobId) {
  const timeoutMs = 20 * 60 * 1000
  const started = Date.now()

  while (Date.now() - started < timeoutMs) {
    const res = await safeFetch(`/api/index/${jobId}`, { method: 'GET' })
    const payload = await res.json()
    if (!res.ok) throw new Error(payload.detail || 'Failed to read index status')

    if (payload.status === 'completed') {
      return payload.result || { indexed_files: 0, indexed_chunks: 0, skipped_files: [] }
    }
    if (payload.status === 'failed') {
      throw new Error(payload.error || 'Index job failed')
    }

    await new Promise((resolve) => setTimeout(resolve, 1500))
  }

  throw new Error('Index job timeout')
}
