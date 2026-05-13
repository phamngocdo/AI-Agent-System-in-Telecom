import React from 'react';
import { useOnboarding } from '../../context/OnboardingContext';

function WelcomeModal() {
  const { nextStep, skipTour } = useOnboarding();

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(15,20,40,0.45)', backdropFilter: 'blur(4px)',
      zIndex: 9998, display: 'flex', alignItems: 'center', justifyContent: 'center',
      animation: 'fadeIn 0.3s ease both'
    }}>
      <div style={{
        background: 'var(--color-surface)', borderRadius: '24px', padding: '40px 36px',
        maxWidth: '480px', width: '90%', boxShadow: '0 24px 64px rgba(0,0,0,0.15)',
        animation: 'fadeUp 0.4s var(--ease-default) both', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center'
      }}>
        <div style={{
          width: '72px', height: '72px', borderRadius: '50%', background: 'rgba(79,142,247,0.1)',
          fontSize: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px'
        }}>
          🤖
        </div>
        
        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 800,
          color: 'var(--color-primary)', marginBottom: '12px', marginTop: 0
        }}>
          Chào mừng bạn! 👋
        </h2>
        
        <p style={{
          color: 'var(--color-muted)', fontSize: '0.95rem', lineHeight: 1.6, marginBottom: '32px'
        }}>
          Hãy để chúng tôi hướng dẫn bạn qua các tính năng chính — chỉ mất khoảng 30 giây.
        </p>

        <div style={{ display: 'flex', gap: '16px' }}>
          <button onClick={nextStep} style={{
            background: 'var(--color-accent)', color: 'white', padding: '12px 28px',
            borderRadius: '999px', fontWeight: 600, border: 'none', cursor: 'pointer',
            transition: 'transform 0.2s, box-shadow 0.2s', fontSize: '1rem'
          }}>
            Bắt đầu tour →
          </button>
          
          <button onClick={skipTour} style={{
            background: 'transparent', color: 'var(--color-muted)', padding: '12px 24px',
            fontWeight: 600, border: 'none', cursor: 'pointer', textDecoration: 'none', fontSize: '1rem'
          }} onMouseOver={(e) => e.target.style.textDecoration = 'underline'} onMouseOut={(e) => e.target.style.textDecoration = 'none'}>
            Bỏ qua
          </button>
        </div>
      </div>
    </div>
  );
}

export default WelcomeModal;
