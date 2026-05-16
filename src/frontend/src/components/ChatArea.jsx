import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { useAppContext } from '../context/AppContext';
import { createUniqueConversationTitle } from '../utils/conversationTitles';

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const MAX_ATTACHMENT_FILES = 5;
const MAX_ATTACHMENT_TOTAL_BYTES = 3 * 1024 * 1024;
const ATTACHMENT_ACCEPT = '.pdf,.md,application/pdf,application/x-pdf,text/markdown,text/plain,text/x-markdown';
const PDF_CONTENT_TYPES = new Set(['', 'application/pdf', 'application/x-pdf', 'application/octet-stream']);
const MARKDOWN_CONTENT_TYPES = new Set(['', 'text/markdown', 'text/plain', 'text/x-markdown', 'application/octet-stream']);

const createMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const markdownRemarkPlugins = [remarkMath, remarkGfm, remarkBreaks];
const markdownRehypePlugins = [[rehypeKatex, { strict: false }]];
const markdownComponents = {
  table: ({ children, node: _node, ...props }) => (
    <div className="markdown-table-wrap">
      <table {...props}>{children}</table>
    </div>
  )
};

const splitLooseTableRow = (line) => {
  const trimmed = line.trim();

  if (!trimmed || trimmed.includes('|')) {
    return null;
  }

  const usesTabs = /\t/.test(trimmed);
  const hasWideSpaces = / {2,}/.test(trimmed);
  if (!usesTabs && !hasWideSpaces) {
    return null;
  }

  const cells = trimmed
    .split(usesTabs ? /\t+/ : / {2,}/)
    .map(cell => cell.trim());

  if (cells.length < 2 || cells.some(cell => !cell)) {
    return null;
  }

  return { cells, separator: usesTabs ? 'tab' : 'space' };
};

const escapeMarkdownTableCell = (cell) => cell.replace(/\|/g, '\\|');

const formatMarkdownTableRow = (cells) => (
  `| ${cells.map(escapeMarkdownTableCell).join(' | ')} |`
);

const formatLooseTable = (rows) => [
  formatMarkdownTableRow(rows[0]),
  formatMarkdownTableRow(rows[0].map(() => '---')),
  ...rows.slice(1).map(formatMarkdownTableRow)
];

const collectLooseTable = (lines, startIndex) => {
  const rows = [];
  const separators = new Set();
  let columnCount = null;
  let index = startIndex;

  while (index < lines.length) {
    const row = splitLooseTableRow(lines[index]);
    if (!row) break;

    if (columnCount === null) {
      columnCount = row.cells.length;
    } else if (row.cells.length !== columnCount) {
      break;
    }

    rows.push(row.cells);
    separators.add(row.separator);
    index += 1;
  }

  const minRows = separators.has('tab') ? 2 : 3;
  if (rows.length < minRows) {
    return null;
  }

  return { rows, nextIndex: index };
};

const normalizeLooseMarkdownTables = (content) => {
  const lines = String(content || '').replace(/\\n/g, '\n').split('\n');
  const normalizedLines = [];
  let fenceMarker = null;
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const fenceMatch = line.match(/^\s*(```|~~~)/);

    if (fenceMatch) {
      const marker = fenceMatch[1];
      fenceMarker = fenceMarker === marker ? null : (fenceMarker || marker);
      normalizedLines.push(line);
      index += 1;
      continue;
    }

    if (!fenceMarker) {
      const table = collectLooseTable(lines, index);
      if (table) {
        if (normalizedLines.length && normalizedLines[normalizedLines.length - 1].trim()) {
          normalizedLines.push('');
        }
        normalizedLines.push(...formatLooseTable(table.rows));
        if (lines[table.nextIndex]?.trim()) {
          normalizedLines.push('');
        }
        index = table.nextIndex;
        continue;
      }
    }

    normalizedLines.push(line);
    index += 1;
  }

  return normalizedLines.join('\n');
};

const MessageMarkdown = ({ content }) => (
  <ReactMarkdown
    remarkPlugins={markdownRemarkPlugins}
    rehypePlugins={markdownRehypePlugins}
    components={markdownComponents}
  >
    {normalizeLooseMarkdownTables(content)}
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

const formatBytes = (bytes) => {
  if (!bytes) return '0KB';
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
};

const getFileExtension = (file) => {
  const name = file?.name || '';
  const dotIndex = name.lastIndexOf('.');
  return dotIndex >= 0 ? name.slice(dotIndex).toLowerCase() : '';
};

const getFileTypeLabel = (file) => {
  const extension = getFileExtension(file);
  if (extension === '.pdf') return 'PDF';
  if (extension === '.md') return 'Markdown';
  return 'Không hỗ trợ';
};

const isValidAttachment = (file) => {
  const extension = getFileExtension(file);
  const contentType = (file?.type || '').toLowerCase();
  if (extension === '.pdf') return PDF_CONTENT_TYPES.has(contentType);
  if (extension === '.md') return MARKDOWN_CONTENT_TYPES.has(contentType);
  return false;
};

const getAttachmentValidationError = (candidateFiles) => {
  if (candidateFiles.length > MAX_ATTACHMENT_FILES) {
    return `Chỉ được đính kèm tối đa ${MAX_ATTACHMENT_FILES} file.`;
  }

  const invalidFile = candidateFiles.find(file => !isValidAttachment(file));
  if (invalidFile) {
    return `Chỉ hỗ trợ file PDF hoặc Markdown: ${invalidFile.name}`;
  }

  const totalSize = candidateFiles.reduce((sum, file) => sum + file.size, 0);
  if (totalSize > MAX_ATTACHMENT_TOTAL_BYTES) {
    return `Tổng dung lượng file tối đa ${formatBytes(MAX_ATTACHMENT_TOTAL_BYTES)}.`;
  }

  return '';
};

const isSameFile = (left, right) => (
  left.name === right.name &&
  left.size === right.size &&
  left.lastModified === right.lastModified
);

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

const readChatStream = async (res, { onContent, onStatus }) => {
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

    if (event === 'status') {
      if (typeof payload.content === 'string' && payload.content) {
        onStatus?.(payload.content);
      }
      return false;
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
    loadConversationMessages,
    loadConversationFiles,
    setConversationActiveFileIds
  } = useAppContext();
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const activeRequestConversationIdRef = useRef(null);
  const [isFileModalOpen, setIsFileModalOpen] = useState(false);
  const [fileModalError, setFileModalError] = useState('');

  const activeConv = conversations.find(c => c.id === activeId);
  const messages = activeConv ? activeConv.messages : [];
  const streamingAiMessage = messages.find(m => m.role === 'ai' && m.streaming);
  const showTypingIndicator = typing && (!streamingAiMessage || !streamingAiMessage.content);
  const filesTotalSize = files.reduce((sum, file) => sum + file.size, 0);
  const sessionFiles = activeConv?.files || [];
  const activeFileIds = Array.isArray(activeConv?.activeFileIds)
    ? activeConv.activeFileIds
    : (activeConv?.file_ids || []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, typing]);

  useEffect(() => {
    if (
      abortControllerRef.current &&
      activeRequestConversationIdRef.current &&
      activeId !== activeRequestConversationIdRef.current
    ) {
      abortControllerRef.current.abort();
    }
  }, [activeId]);

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

  useEffect(() => {
    if (
      !activeConv?.backendId ||
      activeConv.filesLoaded ||
      activeConv.loadingFiles ||
      !(activeConv.file_ids || []).length
    ) {
      return;
    }

    loadConversationFiles(activeConv.id).catch(() => {
      // File filters are optional; failed loading should not block chat.
    });
  }, [
    activeConv?.id,
    activeConv?.backendId,
    activeConv?.filesLoaded,
    activeConv?.loadingFiles,
    activeConv?.file_ids,
    loadConversationFiles
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

  const postChatMessage = async (sessionId, message, token, signal, filesToSend = [], selectedFileIds = []) => {
    if (filesToSend.length) {
      const formData = new FormData();
      formData.append('message', message);
      formData.append('stream', 'true');
      formData.append('temperature', String(llmParams.temp));
      formData.append('top_p', String(llmParams.topP));
      formData.append('top_k', String(llmParams.topK));
      formData.append('think', String(llmParams.think));
      formData.append('file_ids', JSON.stringify(selectedFileIds));
      if (llmParams.memory) {
        formData.append('memory', llmParams.memory);
      }
      filesToSend.forEach(file => formData.append('files', file));

      return fetch(`${API_BASE_URL}/api/chat/${sessionId}/files`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
        signal
      });
    }

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
        memory: llmParams.memory || undefined,
        file_ids: selectedFileIds
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

    const rawMessage = inputRef.current.value.trim();
    const filesToSend = [...files];
    const selectedFileIds = activeConv
      ? (Array.isArray(activeConv.activeFileIds) ? activeConv.activeFileIds : (activeConv.file_ids || []))
      : [];
    const message = rawMessage || (
      filesToSend.length > 1
        ? 'Hãy phân tích và tóm tắt nội dung các tệp này.'
        : 'Hãy phân tích và tóm tắt nội dung tệp này.'
    );
    if (!message) return;

    const attachmentError = getAttachmentValidationError(filesToSend);
    if (attachmentError) {
      setFileModalError(attachmentError);
      setIsFileModalOpen(true);
      return;
    }

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
    setFiles([]);

    appendMessage(conversation.id, {
      role: 'user',
      content: message,
      files: filesToSend.map(file => file.name)
    });
    setTyping(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    activeRequestConversationIdRef.current = conversation.id;
    let resolvedSessionId = conversation.backendId || null;
    let assistantMessageId = null;
    let responseContent = '';
    let analyzingFile = filesToSend.length > 0;

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
      resolvedSessionId = sessionId;
      let res = await postChatMessage(
        sessionId,
        message,
        token,
        abortController.signal,
        filesToSend,
        selectedFileIds,
      );

      if (res.status === 404) {
        const newSessionId = await ensureBackendSession(conversation, message, token, true, abortController.signal);
        resolvedSessionId = newSessionId;
        res = await postChatMessage(
          newSessionId,
          message,
          token,
          abortController.signal,
          filesToSend,
          selectedFileIds,
        );
      }

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      await readChatStream(res, {
        onStatus: (statusText) => {
          if (statusText.includes('Đã phân tích')) {
            analyzingFile = false;
          }
          setAssistantContent(statusText);
        },
        onContent: (chunk) => {
          analyzingFile = false;
          responseContent += chunk;
          setAssistantContent(stripThinking(responseContent));
        }
      });

      if (!responseContent.trim()) {
        setAssistantContent('Model chưa trả về nội dung sau phần thinking.');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        const stoppedMessage = analyzingFile
          ? 'Người dùng đã dừng việc phân tích tệp.'
          : 'Đã dừng tạo câu trả lời.';
        setAssistantContent(stripThinking(responseContent) || stoppedMessage);
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
      if (activeRequestConversationIdRef.current === conversation.id) {
        activeRequestConversationIdRef.current = null;
      }
      if (filesToSend.length && resolvedSessionId) {
        loadConversationFiles(conversation.id, undefined, true, resolvedSessionId).catch(() => {});
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
    setFileModalError('');
  };

  const addSelectedFiles = (fileList) => {
    const selectedFiles = Array.from(fileList || []);
    if (!selectedFiles.length) return;

    const uniqueSelectedFiles = selectedFiles.filter(file => (
      !files.some(existingFile => isSameFile(existingFile, file))
    ));
    const nextFiles = [...files, ...uniqueSelectedFiles];
    const validationError = getAttachmentValidationError(nextFiles);

    if (validationError) {
      setFileModalError(validationError);
      return;
    }

    setFiles(nextFiles);
    setFileModalError('');
  };

  const openFileModal = () => {
    setFileModalError('');
    setIsFileModalOpen(true);
  };

  const handleFileInputChange = (event) => {
    addSelectedFiles(event.target.files);
    event.target.value = '';
  };

  const toggleActiveFile = (fileId) => {
    if (!activeConv) return;
    const nextFileIds = activeFileIds.includes(fileId)
      ? activeFileIds.filter(activeFileId => activeFileId !== fileId)
      : [...activeFileIds, fileId];
    setConversationActiveFileIds(activeConv.id, nextFileIds);
  };

  const selectAllSessionFiles = () => {
    if (!activeConv) return;
    setConversationActiveFileIds(activeConv.id, sessionFiles.map(file => file.file_id));
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
                  <div className="welcome-card-desc">Upload file PDF/Markdown và hỏi nội dung bên trong</div>
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
          {sessionFiles.length > 0 && (
            <div className="session-files-bar">
              <div className="session-files-head">
                <span>Tài liệu tìm kiếm</span>
                <button type="button" onClick={selectAllSessionFiles}>Tất cả</button>
              </div>
              <div className="session-file-list">
                {sessionFiles.map(file => {
                  const checked = activeFileIds.includes(file.file_id);
                  return (
                    <label key={file.file_id} className={`session-file-toggle ${checked ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleActiveFile(file.file_id)}
                      />
                      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                      <span title={file.filename}>{file.filename}</span>
                      <small>{file.chunk_count} đoạn</small>
                    </label>
                  );
                })}
              </div>
            </div>
          )}
          <div className="upload-previews">
            {files.map((f, i) => (
              <div key={i} className="upload-chip">
                <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                {f.name} <span className="chip-size">({formatBytes(f.size)})</span>
                <button className="chip-del" onClick={() => removeFile(i)}>
                  <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
              </div>
            ))}
          </div>
          <div className="input-box">
            <div className="input-tools-row">
              <button
                type="button"
                className="input-tool"
                title="Đính kèm file"
                data-tour="attach-document"
                onClick={openFileModal}
              >
                <svg viewBox="0 0 24 24">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                </svg>
                Đính kèm
              </button>
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

      {isFileModalOpen && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setIsFileModalOpen(false); }}>
          <div className="modal modal-md attach-modal" role="dialog" aria-modal="true">
            <button className="modal-close" onClick={() => setIsFileModalOpen(false)} aria-label="Đóng">
              <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
            </button>
            <div className="modal-header">
              <h2>Đính kèm tài liệu</h2>
              <p>PDF hoặc Markdown, tối đa {MAX_ATTACHMENT_FILES} file và {formatBytes(MAX_ATTACHMENT_TOTAL_BYTES)} tổng.</p>
            </div>

            <div className="attach-picker">
              <input
                ref={fileInputRef}
                className="attach-hidden-input"
                type="file"
                multiple
                accept={ATTACHMENT_ACCEPT}
                onChange={handleFileInputChange}
              />
              <button type="button" className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
                <svg viewBox="0 0 24 24">
                  <path d="M12 5v14" />
                  <path d="M5 12h14" />
                </svg>
                Chọn file
              </button>
              <div className="attach-limit">{files.length}/{MAX_ATTACHMENT_FILES} file · {formatBytes(filesTotalSize)}/{formatBytes(MAX_ATTACHMENT_TOTAL_BYTES)}</div>
            </div>

            {fileModalError && <div className="attach-error">{fileModalError}</div>}

            <div className="attach-table-wrap">
              {files.length > 0 ? (
                <table className="attach-table">
                  <thead>
                    <tr>
                      <th>Tên file</th>
                      <th>Loại</th>
                      <th>Dung lượng</th>
                      <th aria-label="Thao tác"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.map((file, index) => (
                      <tr key={`${file.name}-${file.size}-${file.lastModified}`}>
                        <td>
                          <div className="attach-file-name">
                            <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                            <span title={file.name}>{file.name}</span>
                          </div>
                        </td>
                        <td>{getFileTypeLabel(file)}</td>
                        <td>{formatBytes(file.size)}</td>
                        <td>
                          <button type="button" className="attach-remove-btn" onClick={() => removeFile(index)} aria-label={`Xóa ${file.name}`}>
                            <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="attach-empty">Chưa có file nào.</div>
              )}
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!files.length}
                onClick={() => {
                  setFiles([]);
                  setFileModalError('');
                }}
              >
                Xóa tất cả
              </button>
              <button type="button" className="btn btn-primary" onClick={() => setIsFileModalOpen(false)}>Xong</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default ChatArea;
