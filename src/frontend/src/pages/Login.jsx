import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import '../styles/pages.css';

function Login() {
  const navigate = useNavigate();
  const { login } = useAppContext();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/chat');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-left">
        <div className="auth-left-content">
          <div className="auth-brand">TelcoLLM</div>
          <div className="auth-tagline">Trợ lý thông minh cho mọi tác vụ của bạn.</div>
          <ul className="trust-lines">
            <li>✓ Hỗ trợ phân tích tài liệu</li>
            <li>✓ Tích hợp đa mô hình AI chuyên biệt</li>
          </ul>

          <div className="auth-decoration">
            <div className="chat-bubble user">
              TelcoLLM có thể phân tích báo cáo viễn thông này giúp tôi không?
            </div>
            <div className="chat-bubble ai">
              Tất nhiên. Tôi sẽ đối chiếu số liệu và tóm tắt những điểm cần lưu ý...
            </div>
          </div>
        </div>
      </div>

      <div className="auth-right">
        <div className="form-card">
          <div className="mobile-brand">TelcoLLM</div>
          <h2 className="form-card-title">Đăng nhập</h2>
          <p className="form-card-subtitle">Chào mừng bạn quay trở lại. Vui lòng đăng nhập để tiếp tục.</p>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Ví dụ: user@TelcoLLM.ai"
              />
            </div>
            <div className="form-group">
              <label>Mật khẩu</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Nhập mật khẩu của bạn"
              />
            </div>
            <button type="submit" className="btn-submit" disabled={loading}>
              {loading ? 'Đang xử lý...' : 'Đăng nhập'}
            </button>
          </form>

          <div className="auth-footer">
            Chưa có tài khoản? <Link to="/register">Đăng ký ngay</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
