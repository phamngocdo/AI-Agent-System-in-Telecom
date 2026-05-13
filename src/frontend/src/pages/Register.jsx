import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import '../styles/pages.css';

function Register() {
  const navigate = useNavigate();
  const { registerUser } = useAppContext();

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await registerUser(fullName, email, password);
      navigate('/login');
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
          <div className="auth-tagline">Bắt đầu hành trình cùng trợ lý thông minh.</div>
          <ul className="trust-lines">
            <li>✓ Truy cập không giới hạn</li>
            <li>✓ Dữ liệu an toàn & bảo mật</li>
            <li>✓ Hỗ trợ 24/7</li>
          </ul>

          <div className="auth-decoration">
            <div className="chat-bubble user">
              Tóm tắt tài liệu 50 trang này thành 3 ý chính giúp tôi nhé!
            </div>
            <div className="chat-bubble ai">
              Đang phân tích tài liệu... Dưới đây là 3 điểm mấu chốt quan trọng nhất:
            </div>
          </div>
        </div>
      </div>

      <div className="auth-right">
        <div className="form-card">
          <div className="mobile-brand">TelcoLLM</div>
          <h2 className="form-card-title">Đăng ký tài khoản</h2>
          <p className="form-card-subtitle">Tạo tài khoản mới để trải nghiệm đầy đủ tính năng.</p>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Họ và tên</label>
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Nhập họ tên của bạn"
              />
            </div>
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
                placeholder="Tạo mật khẩu an toàn"
              />
            </div>
            <button type="submit" className="btn-submit" disabled={loading}>
              {loading ? 'Đang xử lý...' : 'Đăng ký'}
            </button>
          </form>

          <div className="auth-footer">
            Đã có tài khoản? <Link to="/login">Đăng nhập</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Register;
