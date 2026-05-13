import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../../context/AppContext';
import { useOnboarding } from '../../context/OnboardingContext';

function SettingsModal({ closeModal }) {
  const { api, setApi, user, setUser, llmParams, setLlmParams, updateProfile, logout } = useAppContext();
  const { startTour, storageKey } = useOnboarding();
  const [activeTab, setActiveTab] = useState('params');
  const navigate = useNavigate();

  const [editName, setEditName] = useState(user.name || '');
  const [editPassword, setEditPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  const handleUpdate = async () => {
    setLoading(true);
    setMsg('');
    try {
      await updateProfile(editName, editPassword);
      setMsg('Cập nhật thành công!');
      setEditPassword('');
    } catch (e) {
      setMsg(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    closeModal();
    navigate('/');
  };

  const handleReplayTour = () => {
    if (storageKey) {
      localStorage.removeItem(storageKey);
    }
    localStorage.removeItem('onboarding_completed');
    closeModal();
    startTour();
  };

  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) closeModal(); }}>
      <div className="modal modal-lg" style={{ position: 'relative' }}>
        <button className="modal-close" onClick={closeModal}>
          <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
        </button>
        <div className="modal-header">
          <h2>Cài đặt</h2>
        </div>
        <div className="settings-tabs">
          <button className={`settings-tab ${activeTab === 'params' ? 'active' : ''}`} onClick={() => setActiveTab('params')}>Phong cách cá nhân</button>
          <button className={`settings-tab ${activeTab === 'profile' ? 'active' : ''}`} onClick={() => setActiveTab('profile')}>Tài khoản</button>
        </div>

        {activeTab === 'params' && (
          <div className="settings-panel active">
            <div className="settings-section">
              <h3>Long-term Memory</h3>

              <div className="field">
                <label>Thông tin cá nhân hóa (Long-term Memory)</label>
                <textarea
                  placeholder="Ví dụ: Tôi là lập trình viên frontend, hãy luôn trả lời bằng code React. Giao tiếp ngắn gọn và xưng 'mình'..."
                  value={llmParams.memory || ''}
                  onChange={e => setLlmParams({ ...llmParams, memory: e.target.value })}
                ></textarea>
                <div className="field-hint">AI sẽ luôn ghi nhớ ngữ cảnh này và áp dụng vào mọi câu trả lời để phù hợp với riêng bạn.</div>
              </div>
            </div>

            <div className="settings-section">
              <h3>Tham số phản hồi</h3>
              <div className="field">
                <label>Độ sáng tạo (Temperature): {llmParams.temp}</label>
                <div className="range-row">
                  <input
                    className="range-control"
                    type="range"
                    min="0" max="1" step="0.1"
                    value={llmParams.temp}
                    onChange={e => setLlmParams({ ...llmParams, temp: parseFloat(e.target.value) })}
                  />
                  <span className="range-value">{llmParams.temp}</span>
                </div>
                <div className="field-hint">
                  {llmParams.temp <= 0.3 ? 'Tính logic cao, câu trả lời chính xác và nhất quán (Range: 0.0 - 0.3)' :
                    llmParams.temp <= 0.7 ? 'Cân bằng giữa logic và sáng tạo (Range: 0.4 - 0.7)' :
                      'Sáng tạo phong phú, ngẫu nhiên cao (Range: 0.8 - 2.0)'}
                </div>
              </div>

              <div className="field">
                <label>Top P: {llmParams.topP}</label>
                <div className="range-row">
                  <input
                    className="range-control"
                    type="range"
                    min="0" max="1" step="0.05"
                    value={llmParams.topP}
                    onChange={e => setLlmParams({ ...llmParams, topP: parseFloat(e.target.value) })}
                  />
                  <span className="range-value">{llmParams.topP}</span>
                </div>
                <div className="field-hint">Giới hạn không gian từ vựng dựa trên xác suất tích lũy. Giá trị càng thấp, AI càng chọn từ ngữ phổ biến.</div>
              </div>

              <div className="field">
                <label>Top K: {llmParams.topK}</label>
                <div className="range-row">
                  <input
                    className="range-control"
                    type="range"
                    min="1" max="100" step="1"
                    value={llmParams.topK}
                    onChange={e => setLlmParams({ ...llmParams, topK: parseInt(e.target.value) })}
                  />
                  <span className="range-value">{llmParams.topK}</span>
                </div>
                <div className="field-hint">Giới hạn số lượng từ có khả năng cao nhất được xem xét ở mỗi bước.</div>
              </div>

            </div>
          </div>
        )}

        {activeTab === 'profile' && (
          <div className="settings-panel active">
            <div className="settings-section">
              <h3>Thông tin tài khoản</h3>
              <div className="field">
                <label>Họ và tên</label>
                <input type="text" value={editName} onChange={e => setEditName(e.target.value)} />
              </div>
              <div className="field">
                <label>Email (Không thể thay đổi)</label>
                <input type="email" value={user.email} disabled />
              </div>
              <div className="field">
                <label>Mật khẩu mới (bỏ trống nếu không đổi)</label>
                <input type="password" value={editPassword} onChange={e => setEditPassword(e.target.value)} placeholder="Nhập mật khẩu mới..." />
              </div>
              {msg && <div style={{ color: msg.includes('thành công') ? 'var(--green)' : 'var(--red)', marginBottom: '12px' }}>{msg}</div>}
              <button onClick={handleUpdate} disabled={loading} style={{ padding: '8px 16px', background: 'var(--accent)', color: 'white', border: 'none', borderRadius: '6px', fontWeight: '500' }}>
                {loading ? 'Đang lưu...' : 'Lưu thay đổi'}
              </button>
            </div>

            <div className="settings-section" style={{ marginTop: '32px', borderTop: '1px solid var(--border)', paddingTop: '24px' }}>
              <h3>Hướng dẫn sử dụng</h3>
              <p style={{ color: 'var(--text3)', fontSize: '0.9rem', marginBottom: '16px' }}>Bạn có thể xem lại hướng dẫn các tính năng chính của ứng dụng.</p>
              <button onClick={handleReplayTour} style={{ padding: '8px 16px', background: 'var(--bg3)', color: 'var(--text)', border: 'none', borderRadius: '6px', fontWeight: '500' }}>
                Xem lại hướng dẫn
              </button>
            </div>

            <div className="settings-section" style={{ marginTop: '32px', borderTop: '1px solid var(--border)', paddingTop: '24px' }}>
              <h3>Đăng xuất</h3>
              <p style={{ color: 'var(--text3)', fontSize: '0.9rem', marginBottom: '16px' }}>Bạn có thể đăng nhập lại bất cứ lúc nào trên thiết bị này.</p>
              <button onClick={handleLogout} style={{ padding: '8px 16px', background: 'var(--red-bg)', color: 'var(--red)', border: 'none', borderRadius: '6px', fontWeight: '600' }}>
                Đăng xuất khỏi tài khoản
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SettingsModal;
