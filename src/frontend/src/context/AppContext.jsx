import { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const normalizeSession = (session) => {
  const id = session.id || session._id;
  return {
    id,
    backendId: id,
    title: session.title || 'Hội thoại đã lưu',
    tag: 'chat',
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

export function AppProvider({ children }) {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [api, setApi] = useState({ url: 'http://localhost:8000/v1', key: '', sysPrompt: '', temp: 0.7, maxTokens: 2048, timeout: 60 });
  const [model, setModel] = useState({ id: 'gpt-4o', label: 'GPT-4o' });
  const [llmParams, setLlmParams] = useState({ temp: 0.7, topP: 1.0, topK: 40, think: false, memory: '' });
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
        setUser({ name: data.full_name || data.email.split('@')[0], email: data.email, loggedIn: true });
      })
      .catch(() => {
        localStorage.removeItem('access_token');
        setUser(null);
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
      setUser({ name: meData.full_name || meData.email.split('@')[0], email: meData.email, loggedIn: true });
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

  const updateProfile = async (fullName, password) => {
    const token = localStorage.getItem('access_token');
    const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ full_name: fullName, password: password || undefined })
    });

    if (!res.ok) {
      throw new Error('Cập nhật thông tin thất bại.');
    }

    const meData = await res.json();
    setUser({ name: meData.full_name || meData.email.split('@')[0], email: meData.email, loggedIn: true });
    return true;
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    setUser(null);
    setAuthLoading(false);
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
    loadConversations, loadConversationMessages,
    renameConversation, deleteConversation,
    login, registerUser, updateProfile, logout
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useAppContext = () => useContext(AppContext);
