import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkBreaks from 'remark-breaks'
import remarkMath from 'remark-math'

type LlmMarkdownProps = {
  text: string
  className?: string
}

/**
 * Приводит сырой Markdown от LLM к виду, который стабильно парсится CommonMark / react-markdown:
 * — «2.1. Заголовок» в начале строки иначе становится вложенным <ol> и ломает вёрстку;
 * — «[34:20]» может трактоваться как link reference;
 * — одиночные переводы строк с remark-breaks дают читаемые переносы внутри блока.
 */
export function normalizeLlmMarkdown(src: string): string {
  let s = src.replace(/\r\n/g, '\n').trimEnd()
  if (!s) return s

  // Временные метки [м:сс] / [ч:мм:сс] — экранируем скобки, чтобы не считались ссылками.
  s = s.replace(/\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g, '\\[$1\\]')

  // Подпункты «2.1.», «3.2.1.» в начале строки — экранируем точки (не список CommonMark).
  s = s.replace(/^(\s*)(\d+(?:\.\d+)+)\.(\s)/gm, (_m, indent: string, nums: string, sp: string) => {
    const escaped = String(nums).replace(/\./g, '\\.') + '\\.'
    return `${indent}${escaped}${sp}`
  })

  // Типичные подписи после абзаца без пустой строки — разделяем блоки.
  s = s.replace(/([^\n])\n(Источник:|Пояснение:|Примечание:)/g, '$1\n\n$2')

  return s
}

/** Рендер текста от LLM (суммаризация, RAG) с поддержкой Markdown и LaTeX ($...$, $$...$$). */
export function LlmMarkdown({ text, className }: LlmMarkdownProps) {
  const raw = (text ?? '').trim()
  if (!raw) return null
  const t = normalizeLlmMarkdown(raw)
  return (
    <div className={className ? `llm-markdown ${className}` : 'llm-markdown'}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkBreaks]}
        rehypePlugins={[
          [
            rehypeKatex,
            {
              strict: 'ignore',
              throwOnError: false,
              errorColor: '#f87171',
            },
          ],
        ]}
      >
        {t}
      </ReactMarkdown>
    </div>
  )
}
