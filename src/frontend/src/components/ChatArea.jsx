import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { useAppContext } from '../context/AppContext';
import { createUniqueConversationTitle } from '../utils/conversationTitles';

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const createMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const markdownRemarkPlugins = [remarkMath, remarkGfm, remarkBreaks];
const markdownRehypePlugins = [[rehypeKatex, { strict: false }]];

const MessageMarkdown = ({ content }) => (
  <ReactMarkdown
    remarkPlugins={markdownRemarkPlugins}
    rehypePlugins={markdownRehypePlugins}
  >
    {String(content || '').replace(/\\n/g, '\n')}
  </ReactMarkdown>
);

const stripThinking = (content) => {
  if (!content) return '';
  let text = String(content);
  if (!/<\/?think\b/i.test(text)) return text;

  const closeMatches = [...text.matchAll(/<\/think\s*>/gi)];
  if (closeMatches.length) {
    const lastClose = closeMatches[closeMatches.length - 1];
    text = text.slice(lastClose.index + lastClose[0].length);
  }
  text = text.replace(/<think\b[^>]*>[\s\S]*?<\/think\s*>/gi, '');
  text = text.replace(/<think\b[^>]*>[\s\S]*$/i, '');
  return text.replace(/<\/think\s*>/gi, '').trimStart();
};

const parseSseEvent = (eventText) => {
  const lines = eventText.split('\n');
  let event = 'message';
  const dataLines = [];

  lines.forEach((line) => {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      const data = line.slice(5);
      dataLines.push(data.startsWith(' ') ? data.slice(1) : data);
    }
  });

  return {
    event,
    data: dataLines.join('\n')
  };
};

const readChatStream = async (res, onContent) => {
  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error('Trình duyệt không hỗ trợ đọc streaming response.');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  const handleEvent = (eventText) => {
    if (!eventText.trim()) return false;

    const { event, data } = parseSseEvent(eventText);
    if (!data) return false;
    if (data === '[DONE]') return true;

    let payload = null;
    try {
      payload = JSON.parse(data);
    } catch {
      payload = { content: data };
    }

    if (event === 'error') {
      throw new Error(payload.error || data || 'Streaming response lỗi.');
    }

    if (typeof payload.content === 'string' && payload.content) {
      onContent(payload.content);
    }

    return false;
  };

  let isDone = false;
  while (!isDone) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let eventBoundary = buffer.indexOf('\n\n');
    while (eventBoundary !== -1) {
      const eventText = buffer.slice(0, eventBoundary);
      buffer = buffer.slice(eventBoundary + 2);
      isDone = handleEvent(eventText);
      if (isDone) break;
      eventBoundary = buffer.indexOf('\n\n');
    }
  }

  buffer += decoder.decode().replace(/\r\n/g, '\n');
  if (!isDone && buffer.trim()) {
    handleEvent(buffer);
  }
};

function ChatArea() {
  const {
    conversations,
    setConversations,
    activeId,
    setActiveId,
    user,
    files,
    setFiles,
    typing,
    setTyping,
    llmParams,
    setLlmParams,
    loadConversationMessages
  } = useAppContext();
  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const activeConv = conversations.find(c => c.id === activeId);
  const messages = activeConv ? activeConv.messages : [];
  const streamingAiMessage = messages.find(m => m.role === 'ai' && m.streaming);
  const showTypingIndicator = typing && (!streamingAiMessage || !streamingAiMessage.content);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, typing]);

  useEffect(() => {
    if (!activeConv?.backendId || activeConv.messagesLoaded || activeConv.loadingMessages) {
      return;
    }

    loadConversationMessages(activeConv.id).catch(() => {
      // The conversation row remains visible; the user can retry by refreshing.
    });
  }, [
    activeConv?.id,
    activeConv?.backendId,
    activeConv?.messagesLoaded,
    activeConv?.loadingMessages,
    loadConversationMessages
  ]);

  const appendMessage = (conversationId, message) => {
    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? {
          ...conv,
          messages: [...conv.messages, message],
          messagesLoaded: true,
          updatedAt: new Date()
        }
        : conv
    )));
  };

  const updateMessage = (conversationId, messageId, updates) => {
    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? {
          ...conv,
          messages: conv.messages.map(msg => (
            msg.id === messageId ? { ...msg, ...updates } : msg
          ))
        }
        : conv
    )));
  };

  const updateMessageContent = (conversationId, messageId, content) => {
    updateMessage(conversationId, messageId, { content });
  };

  const getErrorMessage = async (res) => {
    try {
      const data = await res.json();
      if (typeof data.detail === 'string') return data.detail;
      return data.message || `HTTP ${res.status}`;
    } catch {
      return `HTTP ${res.status}`;
    }
  };

  const ensureBackendSession = async (conversation, message, token, forceNew = false, signal) => {
    if (conversation.backendId && !forceNew) {
      return conversation.backendId;
    }

    const title = conversation.title || createUniqueConversationTitle(conversations);
    const res = await fetch(`${API_BASE_URL}/api/sessions/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ title }),
      signal
    });

    if (!res.ok) {
      throw new Error(await getErrorMessage(res));
    }

    const session = await res.json();
    const backendId = session.id || session._id;

    setConversations(prev => prev.map(conv => (
      conv.id === conversation.id
        ? { ...conv, backendId, title: session.title || title }
        : conv
    )));

    return backendId;
  };

  const postChatMessage = async (sessionId, message, token, signal) => {
    return fetch(`${API_BASE_URL}/api/chat/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        message,
        stream: true,
        temperature: llmParams.temp,
        top_p: llmParams.topP,
        top_k: llmParams.topK,
        think: llmParams.think,
        memory: llmParams.memory || undefined
      }),
      signal
    });
  };

  const stopGeneration = () => {
    abortControllerRef.current?.abort();
  };

  const sendMessage = async () => {
    if (typing) {
      stopGeneration();
      return;
    }

    if (!inputRef.current) return;

    const message = inputRef.current.value.trim();
    if (!message) return;

    const token = localStorage.getItem('access_token');
    if (!token) {
      appendMessage(activeId, { role: 'ai', content: 'Bạn cần đăng nhập để gửi tin nhắn.' });
      return;
    }

    const conversation = activeConv || {
      id: Date.now(),
      title: createUniqueConversationTitle(conversations),
      tag: 'chat',
      createdAt: new Date(),
      updatedAt: new Date(),
      messages: []
    };

    if (!activeConv) {
      setConversations(prev => [conversation, ...prev]);
      setActiveId(conversation.id);
    }

    inputRef.current.value = '';
    inputRef.current.style.height = 'auto';

    appendMessage(conversation.id, { role: 'user', content: message, files: [] });
    setTyping(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    let assistantMessageId = null;
    let responseContent = '';

    const ensureAssistantMessage = () => {
      if (assistantMessageId) return assistantMessageId;

      assistantMessageId = createMessageId();
      appendMessage(conversation.id, {
        id: assistantMessageId,
        role: 'ai',
        content: '',
        streaming: true
      });
      return assistantMessageId;
    };

    const setAssistantContent = (content) => {
      const messageId = ensureAssistantMessage();
      updateMessageContent(conversation.id, messageId, content);
    };

    try {
      const sessionId = await ensureBackendSession(conversation, message, token, false, abortController.signal);
      let res = await postChatMessage(sessionId, message, token, abortController.signal);

      if (res.status === 404) {
        const newSessionId = await ensureBackendSession(conversation, message, token, true, abortController.signal);
        res = await postChatMessage(newSessionId, message, token, abortController.signal);
      }

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      await readChatStream(res, (chunk) => {
        responseContent += chunk;
        setAssistantContent(stripThinking(responseContent));
      });

      if (!responseContent.trim()) {
        setAssistantContent('Model chưa trả về nội dung sau phần thinking.');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setAssistantContent(stripThinking(responseContent) || 'Đã dừng tạo câu trả lời.');
        return;
      }

      setAssistantContent(responseContent
        ? stripThinking(responseContent)
        : `Không gọi được TelcoLLM: ${err.message}`
      );
    } finally {
      if (assistantMessageId) {
        updateMessage(conversation.id, assistantMessageId, { streaming: false });
      }
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
      setTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInputChange = () => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 160) + 'px';
    }
  };

  const removeFile = (index) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  return (
    <>
      <div className="messages-wrap" id="messagesWrap">
        <div className="messages-inner">
          {!messages.length && (
            <div className="welcome">
              <div className="welcome-logo">
                <svg viewBox="0 0 40 40">
                  <path d="M20 4L34 12V28L20 36L6 28V12L20 4Z" />
                </svg>
              </div>
              <h1>Xin chào, tôi là TelcoLLM</h1>
              <p>Một trợ lý thông minh được huấn luyện bởi gần 90 triệu token về telecom.</p>
              <div className="welcome-cards">
                <div className="welcome-card">
                  <div className="welcome-card-icon">📚</div>
                  <div className="welcome-card-title">Hỏi về tài liệu</div>
                  <div className="welcome-card-desc">Upload file PDF, DOCX và hỏi nội dung bên trong</div>
                </div>
                <div className="welcome-card">
                  <div className="welcome-card-icon">🧮</div>
                  <div className="welcome-card-title">Tính toán</div>
                  <div className="welcome-card-desc">Giải bài toán, tính biểu thức, phân tích số liệu và công thức</div>
                </div>
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role === 'user' ? 'user' : 'ai'}`}>
              <div className="msg-avatar-wrap">
                <div className={`msg-avatar-sm ${m.role === 'user' ? 'user' : 'ai'}`}>
                  {m.role === 'user' ? (user.name?.[0] || 'U') : '✦'}
                </div>
              </div>
              <div className="msg-body">
                <div className="msg-sender">{m.role === 'user' ? user.name : 'TelcoLLM'}</div>
                <div className="msg-bubble">
                  {m.files && m.files.map((f, idx) => (
                    <div key={idx} className={`file-attach ${m.role !== 'user' ? 'ai-style' : ''}`}>
                      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                      {f}
                    </div>
                  ))}
                  <MessageMarkdown content={m.content} />
                  {m.sources && m.sources.length > 0 && (
                    <div className="sources">
                      {m.sources.map((s, idx) => (
                        <div key={idx} className="source-chip">
                          <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                          {s}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {showTypingIndicator && (
            <div className="typing-row">
              <div className="msg-avatar-wrap"><div className="msg-avatar-sm ai">✦</div></div>
              <div className="typing-bubble">
                {llmParams.think && <div className="typing-label">Thinking...</div>}
                <div className="typing-dots"><span></span><span></span><span></span></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-area">
        <div className="input-area-inner">
          <div className="upload-previews">
            {files.map((f, i) => (
              <div key={i} className="upload-chip">
                <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                {f.name} <span className="chip-size">({Math.round(f.size / 1024)}KB)</span>
                <button className="chip-del" onClick={() => removeFile(i)}>
                  <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
              </div>
            ))}
          </div>
          <div className="input-box">
            <div className="input-tools-row">
              <label className="input-tool" title="Đính kèm file" data-tour="attach-document">
                <svg viewBox="0 0 24 24">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                </svg>
                Đính kèm
                <input type="file" multiple style={{ display: 'none' }} onChange={(e) => setFiles([...files, ...e.target.files])} />
              </label>
              <div className="input-tools-sep"></div>
              <button className={`input-tool ${llmParams.think ? 'active' : ''}`} onClick={() => setLlmParams({ ...llmParams, think: !llmParams.think })} data-tour="reasoning-toggle">
                <svg viewBox="0 0 24 24">
                  <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm-1-8a1 1 0 1 1 2 0v2h-2z" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                </svg>
                Think
              </button>
              {llmParams.think && (
                <span className="think-note">Sử dụng khi cần giải quyết vấn đề phức tạp.</span>
              )}

            </div>
            <div className="input-row">
              <textarea
                data-tour="send-message"
                ref={inputRef}
                placeholder="Nhắn gì đó... (Enter gửi, Shift+Enter xuống dòng)"
                rows="1"
                onKeyDown={handleKeyDown}
                onChange={handleInputChange}
              />
              <button
                className={`send-btn ${typing ? 'stop' : ''}`}
                onClick={sendMessage}
                aria-label={typing ? 'Dừng tạo câu trả lời' : 'Gửi tin nhắn'}
                title={typing ? 'Dừng tạo câu trả lời' : 'Gửi tin nhắn'}
              >
                <svg viewBox="0 0 24 24">
                  {typing ? (
                    <rect x="6" y="6" width="12" height="12" />
                  ) : (
                    <>
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </>
                  )}
                </svg>
              </button>
            </div>
          </div>
          <div className="llm-disclaimer">
            TelcoLLM có thể bịa thông tin. Hãy prompt rõ ngữ cảnh và kiểm chứng các câu trả lời quan trọng.
          </div>
        </div>
      </div>
    </>
  );
}

export default ChatArea;
