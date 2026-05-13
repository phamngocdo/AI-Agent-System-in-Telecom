import React from 'react';

function KBModal({ closeModal }) {
  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) closeModal(); }}>
      <div className="modal modal-md" style={{position:'relative'}}>
        <button className="modal-close" onClick={closeModal}>
          <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
        <div className="modal-header">
          <h2>Knowledge Base</h2>
          <p>Upload tài liệu để chatbot tham chiếu khi trả lời</p>
        </div>
        <div style={{border:'2px dashed var(--border2)', borderRadius:'var(--r-lg)', padding:'40px', textAlign:'center', cursor:'pointer'}}>
          <div style={{fontSize:'32px', marginBottom:'10px'}}>📂</div>
          <div>Kéo thả file vào đây</div>
        </div>
      </div>
    </div>
  );
}

export default KBModal;
