import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import '../styles/pages.css';

function Home() {
  const navigate = useNavigate();
  const { user, authLoading } = useAppContext();

  const handleDiscover = () => {
    if (authLoading) {
      return;
    }

    if (user && user.loggedIn) {
      navigate('/chat');
    } else {
      navigate('/login');
    }
  };

  return (
    <div className="home-container" style={{
      background: 'linear-gradient(to bottom, #f0f4fd 0%, #ffffff 100%)',
      justifyContent: 'center',
      alignItems: 'center',
      textAlign: 'center',
      padding: '24px'
    }}>
      <div className="home-centered-content" style={{
        maxWidth: '800px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        animation: 'fadeUp 0.8s var(--ease-default) both'
      }}>

        {/* App Icon matching the reference */}
        <div className="home-app-icon" style={{
          width: '110px',
          height: '110px',
          borderRadius: '28px',
          background: '#4F8EF7',
          boxShadow: '0 20px 40px rgba(79, 142, 247, 0.26)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '32px'
        }}>
          <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '75%', height: '75%' }}>
            <path d="M50 22L74 36V64L50 78L26 64V36L50 22Z" fill="white" />
          </svg>
        </div>

        <h1 className="home-title" style={{
          fontSize: 'clamp(4rem, 8vw, 6.5rem)',
          fontWeight: '500',
          letterSpacing: '-0.04em',
          color: '#111827',
          marginBottom: '28px',
          fontFamily: 'var(--font-display)'
        }}>
          TelcoLLM
        </h1>

        <p className="home-subtitle" style={{
          fontSize: 'clamp(1.1rem, 2.5vw, 1.45rem)',
          color: '#374151',
          lineHeight: '1.6',
          maxWidth: '780px',
          marginBottom: '56px',
          fontWeight: '400',
          fontFamily: 'var(--font-body)'
        }}>
          Xin giới thiệu chatbot hỏi đáp chuyên biệt cho lĩnh vực viễn thông,
          tích hợp mô hình ngôn ngữ được huấn luyện trên dữ liệu miền viễn thông,
          hỗ trợ tra cứu kiến thức và trả lời câu hỏi dựa trên tài liệu.
        </p>

        <button
          className="btn-black-pill"
          onClick={handleDiscover}
          style={{
            background: '#000000',
            color: '#ffffff',
            padding: '16px 32px',
            borderRadius: '999px',
            fontSize: '1.15rem',
            fontWeight: '500',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: 'pointer',
            transition: 'transform 0.2s, box-shadow 0.2s'
          }}
          disabled={authLoading}
          onMouseOver={(e) => {
            if (authLoading) return;
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 12px 24px rgba(0,0,0,0.15)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          {authLoading ? 'Đang kiểm tra đăng nhập...' : 'Khám phá TelcoLLM'}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="7" y1="17" x2="17" y2="7"></line>
            <polyline points="7 7 17 7 17 17"></polyline>
          </svg>
        </button>

      </div>
    </div>
  );
}

export default Home;
