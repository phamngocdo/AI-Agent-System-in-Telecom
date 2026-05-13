import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { createUniqueConversationTitle } from '../utils/conversationTitles';

function Sidebar({ collapsed, mobileOpen, toggleSidebar, openSettings }) {
  const {
    user,
    conversations,
    activeId,
    setActiveId,
    setConversations,
    renameConversation,
    deleteConversation
  } = useAppContext();
  const [filter, setFilter] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [pendingId, setPendingId] = useState(null);

  const filtered = conversations
    .filter(c => c.title.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt));

  const newChat = () => {
    const id = Date.now();
    const now = new Date();
    const title = createUniqueConversationTitle(conversations);
    setConversations([
      {
        id,
        title,
        tag: 'chat',
        createdAt: now,
        updatedAt: now,
        messages: [],
        messagesLoaded: true
      },
      ...conversations
    ]);
    setActiveId(id);
  };

  const startRename = (event, conversation) => {
    event.stopPropagation();
    setEditingId(conversation.id);
    setEditingTitle(conversation.title);
  };

  const cancelRename = (event) => {
    event.stopPropagation();
    setEditingId(null);
    setEditingTitle('');
  };

  const saveRename = async (event, conversation) => {
    event.preventDefault();
    event.stopPropagation();

    const title = editingTitle.trim();
    if (!title || title === conversation.title) {
      setEditingId(null);
      setEditingTitle('');
      return;
    }

    setPendingId(conversation.id);
    try {
      await renameConversation(conversation.id, title);
      setEditingId(null);
      setEditingTitle('');
    } catch (error) {
      window.alert(`Không đổi tên được hội thoại: ${error.message}`);
    } finally {
      setPendingId(null);
    }
  };

  const handleDelete = async (event, conversation) => {
    event.stopPropagation();
    const confirmed = window.confirm(`Xóa "${conversation.title}" và toàn bộ tin nhắn trong hội thoại này?`);
    if (!confirmed) return;

    setPendingId(conversation.id);
    try {
      await deleteConversation(conversation.id);
      if (editingId === conversation.id) {
        setEditingId(null);
        setEditingTitle('');
      }
    } catch (error) {
      window.alert(`Không xóa được hội thoại: ${error.message}`);
    } finally {
      setPendingId(null);
    }
  };

  const getGroupedConvs = () => {
    const today = [], yesterday = [], older = [];
    const now = Date.now();
    filtered.forEach(c => {
      const diff = now - new Date(c.updatedAt || c.createdAt).getTime();
      if (diff < 86400000) today.push(c);
      else if (diff < 172800000) yesterday.push(c);
      else older.push(c);
    });
    return { today, yesterday, older };
  };

  const { today, yesterday, older } = getGroupedConvs();

  const renderGroup = (label, items) => {
    if (!items.length) return null;
    const icons = { rag: '📚', code: '💻', doc: '📄', chat: '💬' };
    const tagLabels = { rag: 'RAG', code: 'Code', doc: 'Doc', chat: 'Chat' };

    return (
      <div className="conv-group">
        <div className="conv-group-label">{label}</div>
        {items.map(c => (
          <div
            key={c.id}
            className={`conv-item ${c.id === activeId ? 'active' : ''}`}
            onClick={() => setActiveId(c.id)}
          >
            <div className="conv-item-icon">{icons[c.tag] || '💬'}</div>
            <div className="conv-item-body">
              {editingId === c.id ? (
                <form className="conv-rename-form" onSubmit={(event) => saveRename(event, c)} onClick={(event) => event.stopPropagation()}>
                  <input
                    value={editingTitle}
                    onChange={(event) => setEditingTitle(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Escape') {
                        cancelRename(event);
                      }
                    }}
                    autoFocus
                    disabled={pendingId === c.id}
                  />
                  <button type="submit" className="conv-rename-btn" title="Lưu tên" disabled={pendingId === c.id}>
                    <svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12" /></svg>
                  </button>
                  <button type="button" className="conv-rename-btn" title="Hủy" onClick={cancelRename} disabled={pendingId === c.id}>
                    <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                  </button>
                </form>
              ) : (
                <>
                  <div className="conv-title">{c.title}</div>
                  <div className="conv-meta">
                    <span className={`tag ${c.tag}`}>{tagLabels[c.tag] || 'Chat'}</span>
                    {c.messagesLoaded ? `${c.messages.length} tin nhắn` : 'Đã lưu'}
                  </div>
                </>
              )}
            </div>
            <div className="conv-actions">
              <button
                className="conv-action-btn"
                title="Đổi tên"
                onClick={(event) => startRename(event, c)}
                disabled={pendingId === c.id}
              >
                <svg viewBox="0 0 24 24"><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" /></svg>
              </button>
              <button
                className="conv-action-btn danger"
                title="Xóa"
                onClick={(event) => handleDelete(event, c)}
                disabled={pendingId === c.id}
              >
                <svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6" /><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><line x1="10" y1="11" x2="10" y2="17" /><line x1="14" y1="11" x2="14" y2="17" /></svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
    {collapsed && !mobileOpen && (
      <button className="sidebar-open-btn" onClick={toggleSidebar} aria-label="Mở sidebar" title="Mở sidebar">
        <svg viewBox="0 0 24 24">
          <line x1="3" y1="6" x2="21" y2="6" stroke="currentColor" />
          <line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" />
          <line x1="3" y1="18" x2="21" y2="18" stroke="currentColor" />
        </svg>
      </button>
    )}
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`} id="sidebar">
      <div className="sidebar-top">
        <div className="sidebar-header-row">
          <div className="logo">
            <div className="logo-mark">
              <svg viewBox="0 0 20 20">
                <path d="M10 2L17 6V14L10 18L3 14V6L10 2Z" fill="white" />
              </svg>
            </div>
            <span className="logo-text">TelcoLLM <span>AI</span></span>
          </div>
          <button className="sidebar-close-btn" onClick={toggleSidebar} aria-label="Đóng sidebar" title="Đóng sidebar">
            <svg viewBox="0 0 24 24">
              <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" />
              <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" />
            </svg>
          </button>
        </div>
        <button className="new-chat-btn" onClick={newChat}>
          <svg viewBox="0 0 24 24">
            <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" />
            <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" />
          </svg>
          Cuộc hội thoại mới
        </button>
      </div>
      <div className="sidebar-search">
        <div className="search-wrap">
          <svg viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="8" stroke="currentColor" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="currentColor" />
          </svg>
          <input
            className="search-input"
            type="text"
            placeholder="Tìm kiếm hội thoại..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>
      <div className="conv-list">
        {!filtered.length && (
          <div className="empty-state">
            <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
            <p>Không tìm thấy hội thoại</p>
          </div>
        )}
        {renderGroup('Hôm nay', today)}
        {renderGroup('Hôm qua', yesterday)}
        {renderGroup('Trước đó', older)}
      </div>
      <div className="sidebar-footer">
        <div className="user-card" onClick={openSettings} data-tour="personalization">
          <div className="avatar">
            {user.name ? user.name[0].toUpperCase() : 'U'}
          </div>
          <div className="user-info">
            <div className="user-name">{user.name}</div>
            <div className="user-email">{user.email || 'Chưa đăng nhập'}</div>
          </div>
          <button className="user-menu-btn">
            <svg viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="1" stroke="currentColor" />
              <circle cx="12" cy="5" r="1" stroke="currentColor" />
              <circle cx="12" cy="19" r="1" stroke="currentColor" />
            </svg>
          </button>
        </div>
      </div>
    </aside>
    </>
  );
}

export default Sidebar;
