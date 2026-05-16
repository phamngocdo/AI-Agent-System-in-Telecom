import { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const DEFAULT_LLM_PARAMS = { temp: 0.7, topP: 1.0, topK: 20, think: false, memory: '' };

const normalizeUser = (data) => ({
  id: data.id || data._id,
  name: data.full_name || data.email.split('@')[0],
  fullName: data.full_name || '',
  email: data.email,
  personalContext: data.personal_context || '',
  loggedIn: true
});

const normalizeSession = (session) => {
  const id = session.id || session._id;
  const fileIds = session.file_ids || [];
  return {
    id,
    backendId: id,
    title: session.title || 'Hội thoại đã lưu',
    tag: 'chat',
    file_ids: fileIds,
    activeFileIds: fileIds,
    files: [],
    filesLoaded: false,
    loadingFiles: false,
    createdAt: session.created_at ? new Date(session.created_at) : new Date(),
    updatedAt: session.updated_at ? new Date(session.updated_at) : new Date(),
    messages: [],
    messagesLoaded: false,
    loadingMessages: false
  };
};

const normalizeMessage = (message) => ({
  id: message.id || message._id,
  role: message.role,
  content: message.content || '',
  file_ids: message.file_ids || null,
  createdAt: message.created_at ? new Date(message.created_at) : new Date()
});

const normalizeChatFile = (file) => ({
  file_id: file.file_id,
  filename: file.filename || 'Tài liệu',
  file_type: file.file_type || '',
  status: file.status || 'unknown',
  chunk_count: file.chunk_count || 0,
  createdAt: file.created_at ? new Date(file.created_at) : null,
  updatedAt: file.updated_at ? new Date(file.updated_at) : null
});

const dedupeIds = (ids) => {
  const seen = new Set();
  const result = [];
  ids.forEach(id => {
    const normalized = String(id || '').trim();
    if (normalized && !seen.has(normalized)) {
      seen.add(normalized);
      result.push(normalized);
    }
  });
  return result;
};

export function AppProvider({ children }) {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [api, setApi] = useState({ url: 'http://localhost:8000/v1', key: '', sysPrompt: '', temp: 0.7, maxTokens: 2048, timeout: 60 });
  const [model, setModel] = useState({ id: 'gpt-4o', label: 'GPT-4o' });
  const [llmParams, setLlmParams] = useState(DEFAULT_LLM_PARAMS);
  const [rag, setRag] = useState({ enabled: true, webSearch: false });
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [typing, setTyping] = useState(false);
  const [files, setFiles] = useState([]);

  const getErrorMessage = async (res) => {
    try {
      const data = await res.json();
      if (typeof data.detail === 'string') return data.detail;
      return data.message || `HTTP ${res.status}`;
    } catch {
      return `HTTP ${res.status}`;
    }
  };

  const loadConversations = async (token = localStorage.getItem('access_token')) => {
    if (!token) return [];

    const res = await fetch(`${API_BASE_URL}/api/sessions/`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!res.ok) {
      throw new Error(await getErrorMessage(res));
    }

    const sessions = await res.json();
    const loadedConversations = sessions.map(normalizeSession);

    setConversations(prev => {
      const localDrafts = prev.filter(conv => !conv.backendId);
      return [...localDrafts, ...loadedConversations];
    });

    setActiveId(prev => {
      if (prev && loadedConversations.some(conv => conv.id === prev)) {
        return prev;
      }
      return loadedConversations[0]?.id || null;
    });

    return loadedConversations;
  };

  const loadConversationMessages = async (
    conversationId,
    token = localStorage.getItem('access_token')
  ) => {
    if (!token || !conversationId) return [];

    const conversation = conversations.find(conv => conv.id === conversationId);
    const backendId = conversation?.backendId || conversationId;
    if (!backendId) return [];

    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? { ...conv, loadingMessages: true }
        : conv
    )));

    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${backendId}/messages`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      const data = await res.json();
      const messages = data.map(normalizeMessage);

      setConversations(prev => prev.map(conv => (
        conv.id === conversationId
          ? { ...conv, messages, messagesLoaded: true, loadingMessages: false }
          : conv
      )));

      return messages;
    } catch (error) {
      setConversations(prev => prev.map(conv => (
        conv.id === conversationId
          ? { ...conv, messagesLoaded: true, loadingMessages: false, loadError: error.message }
          : conv
      )));
      throw error;
    }
  };

  const loadConversationFiles = async (
    conversationId,
    token = localStorage.getItem('access_token'),
    activateNewFiles = false,
    backendIdOverride = null
  ) => {
    if (!token || !conversationId) return [];

    const conversation = conversations.find(conv => conv.id === conversationId);
    const backendId = backendIdOverride || conversation?.backendId || conversationId;
    if (!backendId) return [];

    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? { ...conv, loadingFiles: true }
        : conv
    )));

    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${backendId}/files`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      const data = await res.json();
      const loadedFiles = data.map(normalizeChatFile);
      const fileIds = loadedFiles.map(file => file.file_id);

      setConversations(prev => prev.map(conv => {
        if (conv.id !== conversationId) return conv;

        const previousFileIds = conv.file_ids || [];
        const previousActive = Array.isArray(conv.activeFileIds)
          ? conv.activeFileIds.filter(fileId => fileIds.includes(fileId))
          : fileIds;
        const newFileIds = fileIds.filter(fileId => !previousFileIds.includes(fileId));
        const activeFileIds = activateNewFiles
          ? dedupeIds([...previousActive, ...newFileIds])
          : previousActive;

        return {
          ...conv,
          files: loadedFiles,
          file_ids: fileIds,
          activeFileIds,
          filesLoaded: true,
          loadingFiles: false
        };
      }));

      return loadedFiles;
    } catch (error) {
      setConversations(prev => prev.map(conv => (
        conv.id === conversationId
          ? { ...conv, filesLoaded: true, loadingFiles: false, filesLoadError: error.message }
          : conv
      )));
      throw error;
    }
  };

  const setConversationActiveFileIds = (conversationId, activeFileIds) => {
    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? { ...conv, activeFileIds: dedupeIds(activeFileIds) }
        : conv
    )));
  };

  const renameConversation = async (
    conversationId,
    title,
    token = localStorage.getItem('access_token')
  ) => {
    const nextTitle = title.trim();
    if (!nextTitle) {
      throw new Error('Tên hội thoại không được để trống.');
    }

    const conversation = conversations.find(conv => conv.id === conversationId);
    if (!conversation) {
      throw new Error('Không tìm thấy hội thoại.');
    }

    if (!conversation.backendId) {
      setConversations(prev => prev.map(conv => (
        conv.id === conversationId
          ? { ...conv, title: nextTitle, updatedAt: new Date() }
          : conv
      )));
      return { ...conversation, title: nextTitle };
    }

    if (!token) {
      throw new Error('Bạn cần đăng nhập để đổi tên hội thoại.');
    }

    const res = await fetch(`${API_BASE_URL}/api/sessions/${conversation.backendId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ title: nextTitle })
    });

    if (!res.ok) {
      throw new Error(await getErrorMessage(res));
    }

    const session = await res.json();
    const updatedAt = session.updated_at ? new Date(session.updated_at) : new Date();

    setConversations(prev => prev.map(conv => (
      conv.id === conversationId
        ? { ...conv, title: session.title || nextTitle, updatedAt }
        : conv
    )));

    return session;
  };

  const deleteConversation = async (
    conversationId,
    token = localStorage.getItem('access_token')
  ) => {
    const conversation = conversations.find(conv => conv.id === conversationId);
    if (!conversation) return;

    if (conversation.backendId) {
      if (!token) {
        throw new Error('Bạn cần đăng nhập để xóa hội thoại.');
      }

      const res = await fetch(`${API_BASE_URL}/api/sessions/${conversation.backendId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }
    }

    const remaining = conversations
      .filter(conv => conv.id !== conversationId)
      .sort((a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt));

    setConversations(prev => prev.filter(conv => conv.id !== conversationId));
    setActiveId(current => (
      current === conversationId ? (remaining[0]?.id || null) : current
    ));
  };

  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem('access_token');

    if (!token) {
      setAuthLoading(false);
      return;
    }

    fetch(`${API_BASE_URL}/api/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
      .then(res => {
        if (res.ok) {
          return res.json();
        }
        throw new Error('Invalid token');
      })
      .then(data => {
        const nextUser = normalizeUser(data);
        setUser(nextUser);
        setLlmParams(prev => ({ ...prev, memory: nextUser.personalContext }));
      })
      .catch(() => {
        localStorage.removeItem('access_token');
        setUser(null);
        setLlmParams(prev => ({ ...prev, memory: '' }));
      })
      .finally(() => {
        setAuthLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!user?.loggedIn) return;

    loadConversations().catch(() => {
      setConversations([]);
      setActiveId(null);
    });
  }, [user?.loggedIn]);

  const login = async (email, password) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString()
    });

    if (!res.ok) {
      throw new Error('Đăng nhập thất bại. Vui lòng kiểm tra lại email/mật khẩu.');
    }

    const data = await res.json();
    localStorage.setItem('access_token', data.access_token);

    // Fetch user info
    const meRes = await fetch(`${API_BASE_URL}/api/auth/me`, {
      headers: {
        'Authorization': `Bearer ${data.access_token}`
      }
    });

    if (meRes.ok) {
      const meData = await meRes.json();
      const nextUser = normalizeUser(meData);
      setUser(nextUser);
      setLlmParams(prev => ({ ...prev, memory: nextUser.personalContext }));
      setConversations([]);
      setActiveId(null);
      return;
    }

    localStorage.removeItem('access_token');
    setUser(null);
    throw new Error('Đăng nhập thành công nhưng không lấy được thông tin tài khoản.');
  };

  const registerUser = async (fullName, email, password) => {
    const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, full_name: fullName, password })
    });

    if (!res.ok) {
      throw new Error('Đăng ký thất bại. Email có thể đã tồn tại.');
    }

    return true;
  };

  const updateProfile = async (updatesOrFullName, password, personalContext) => {
    const updates = typeof updatesOrFullName === 'object' && updatesOrFullName !== null
      ? updatesOrFullName
      : { fullName: updatesOrFullName, password, personalContext };
    const payload = {};

    if (Object.prototype.hasOwnProperty.call(updates, 'fullName')) {
      payload.full_name = updates.fullName;
    }
    if (updates.password) {
      payload.password = updates.password;
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'personalContext')) {
      payload.personal_context = updates.personalContext || '';
    }

    const token = localStorage.getItem('access_token');
    const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error('Cập nhật thông tin thất bại.');
    }

    const meData = await res.json();
    const nextUser = normalizeUser(meData);
    setUser(nextUser);
    if (Object.prototype.hasOwnProperty.call(updates, 'personalContext')) {
      setLlmParams(prev => ({ ...prev, memory: nextUser.personalContext }));
    }
    return true;
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    setUser(null);
    setAuthLoading(false);
    setLlmParams(prev => ({ ...prev, memory: '' }));
    setConversations([]);
    setActiveId(null);
  };


  const value = {
    user, setUser,
    authLoading,
    api, setApi,
    model, setModel,
    llmParams, setLlmParams,
    rag, setRag,
    conversations, setConversations,
    activeId, setActiveId,
    typing, setTyping,
    files, setFiles,
    loadConversations, loadConversationMessages, loadConversationFiles,
    setConversationActiveFileIds,
    renameConversation, deleteConversation,
    login, registerUser, updateProfile, logout
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useAppContext = () => useContext(AppContext);
