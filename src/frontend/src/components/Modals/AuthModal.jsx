import React from 'react';

function AuthModal({ closeModal }) {
  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) closeModal(); }}>
      <div className="modal modal-sm" style={{ position: 'relative' }}>
        <button className="modal-close" onClick={closeModal}>
          <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
        </button>
        <div className="modal-header">
          <h2>Đăng nhập</h2>
          <p>Chào mừng trở lại TelcoLLM</p>
        </div>
        <button className="oauth-btn">
          <span className="oauth-icon">G</span> Tiếp tục với Google
        </button>
      </div>
    </div>
  );
}

export default AuthModal;
