import { ChevronDown, FileText, Sparkles, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { filterCitedSources } from '../../services/chat'
import type { ChatMessage, Source } from '../../types/api'

function safeExternalUrl(value: string | null | undefined): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.username || url.password) return null
    return url.toString()
  } catch {
    return null
  }
}

function SourceCard({ source }: { source: Source }) {
  const sourceUrl = source.source_type === 'web' ? safeExternalUrl(source.url) : null
  return <div className="source-card">
    <div className="source-title"><span className="source-number">{source.citation}</span><FileText size={15} />{sourceUrl ? <a href={sourceUrl} target="_blank" rel="noopener noreferrer"><strong>{source.title || source.filename}</strong></a> : <strong>{source.filename}</strong>}</div>
    <div className="source-meta">{source.source_type === 'web' ? <span>网页来源</span> : <span>片段 {source.chunk_index + 1}</span>}{source.published_date && <span>{source.published_date}</span>}{source.page_number != null && <span>第 {source.page_number} 页</span>}{source.similarity != null && <span>相似度 {source.similarity.toFixed(2)}</span>}</div>
    <p>{source.content}</p>
  </div>
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const assistant = message.role === 'assistant'
  const citedSources = assistant ? filterCitedSources(message.content, message.sources) : []
  return <article className={`message-row ${assistant ? 'assistant-row' : 'user-row'}`}>
    <div className={`message-avatar ${assistant ? 'ai-avatar' : 'human-avatar'}`}>{assistant ? <Sparkles size={19} /> : <User size={16} />}</div>
    <div className="message-body"><div className="message-author">{assistant ? 'NodAgent' : '你'}</div>
      {message.content ? <div className="message-content markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children }) => {
        const safeHref = safeExternalUrl(href)
        return safeHref ? <a href={safeHref} target="_blank" rel="noopener noreferrer">{children}</a> : <>{children}</>
      } }}>{message.content}</ReactMarkdown></div> : message.pending && !message.interrupt ? <div className="typing"><span /><span /><span /></div> : null}
      {message.status && <div className="message-status" role="status">{message.status}</div>}
      {message.error && <div className="message-error" role="alert">{message.error}</div>}
      {message.pending && message.content && !message.interrupt && <span className="stream-cursor" aria-label="正在生成" />}
      {citedSources.length > 0 && <details className="sources"><summary><FileText size={15} /> 引用来源（{citedSources.length}） <ChevronDown size={15} className="source-chevron" /></summary><div className="source-list">{citedSources.map(source => <SourceCard key={source.chunk_id} source={source} />)}</div></details>}
    </div>
  </article>
}
