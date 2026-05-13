import React, { useState, useEffect, useCallback } from 'react';
import { useOnboarding } from '../../context/OnboardingContext';

const TOUR_STEPS = [
  { id: "send-message", icon: "💬", title: "Nhắn tin với AI", desc: "Nhập câu hỏi hoặc yêu cầu của bạn vào đây và nhấn gửi để bắt đầu cuộc trò chuyện." },
  { id: "attach-document", icon: "📎", title: "Gắn tài liệu", desc: "Tải lên file PDF, Word hoặc văn bản để AI phân tích và trả lời dựa trên nội dung tài liệu của bạn." },
  { id: "reasoning-toggle", icon: "🧠", title: "Bật Think", desc: "Bật chế độ Think cho câu hỏi cần suy luận kỹ hơn. Khi model đang suy nghĩ, giao diện sẽ hiển thị Thinking và chỉ in phần trả lời cuối cùng." },
  { id: "personalization", icon: "✨", title: "Cá nhân hóa", desc: "Điều chỉnh phong cách phản hồi, ngôn ngữ và các tùy chọn cá nhân để AI phù hợp hơn với bạn." }
];

function TourTooltip() {
  const { currentStep, nextStep, prevStep, skipTour } = useOnboarding();
  const stepIndex = currentStep - 1; // currentStep 1 is index 0
  const stepData = TOUR_STEPS[stepIndex];

  const [rect, setRect] = useState(null);

  const updatePosition = useCallback(() => {
    if (!stepData) return;
    const el = document.querySelector(`[data-tour="${stepData.id}"]`);
    if (el) {
      const r = el.getBoundingClientRect();
      setRect({
        top: r.top - 8,
        left: r.left - 8,
        width: r.width + 16,
        height: r.height + 16,
        bottom: r.bottom + 8,
      });
    } else {
      setRect(null);
    }
  }, [stepData]);

  useEffect(() => {
    updatePosition();
    window.addEventListener('resize', updatePosition);
    // Fallback to update position periodically for dynamic layout shifts
    const interval = setInterval(updatePosition, 100);
    return () => {
      window.removeEventListener('resize', updatePosition);
      clearInterval(interval);
    };
  }, [updatePosition]);

  if (!stepData) return null;

  // Tooltip position calculation
  let tooltipStyle = {};
  if (rect) {
    const isMobile = window.innerWidth < 768;
    if (isMobile) {
      tooltipStyle = {
        bottom: '24px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: 'calc(100vw - 48px)'
      };
    } else {
      // Prefer below if enough space, else above
      const spaceBelow = window.innerHeight - rect.bottom;
      if (spaceBelow > 200) {
        tooltipStyle = {
          top: rect.bottom + 16,
          left: rect.left,
        };
      } else {
        tooltipStyle = {
          bottom: window.innerHeight - rect.top + 16,
          left: rect.left,
        };
      }

      // Ensure it doesn't overflow the right edge
      if (rect.left + 320 > window.innerWidth) {
        tooltipStyle.left = 'auto';
        tooltipStyle.right = '24px';
      }
    }
  }

  return (
    <>
      <div style={{
        position: 'fixed',
        border: '2.5px solid var(--color-accent)',
        borderRadius: '12px',
        boxShadow: '0 0 0 9999px rgba(15,20,40,0.5), 0 0 0 4px rgba(79,142,247,0.2)',
        transition: 'all 0.35s var(--ease-default)',
        pointerEvents: 'none',
        zIndex: 9997,
        ...(rect ? { top: rect.top, left: rect.left, width: rect.width, height: rect.height } : { opacity: 0 })
      }} />

      {rect && (
        <div style={{
          position: 'fixed',
          zIndex: 9999,
          background: 'var(--color-surface)',
          borderRadius: '16px',
          padding: '20px 24px',
          maxWidth: '320px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.14)',
          animation: 'fadeUp 0.3s var(--ease-default) both',
          ...tooltipStyle
        }}>
          <button
            onClick={skipTour}
            style={{ position: 'absolute', top: '16px', right: '16px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)', fontSize: '1rem' }}
          >
            ✕
          </button>

          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-accent)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
            Bước {currentStep} / {TOUR_STEPS.length}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span style={{ fontSize: '1.2rem' }}>{stepData.icon}</span>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-primary)', margin: 0 }}>
              {stepData.title}
            </h3>
          </div>

          <p style={{ fontSize: '0.875rem', color: 'var(--color-muted)', lineHeight: 1.55, marginTop: '6px', marginBottom: '20px' }}>
            {stepData.desc}
          </p>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', gap: '6px' }}>
              {TOUR_STEPS.map((_, i) => (
                <div key={i} style={{
                  width: '8px', height: '8px', borderRadius: '50%',
                  background: i === stepIndex ? 'var(--color-accent)' : 'var(--color-border)',
                  transition: 'background 0.2s'
                }} />
              ))}
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              {currentStep > 1 && (
                <button onClick={prevStep} style={{
                  background: 'transparent', color: 'var(--color-muted)', border: 'none', fontWeight: 600, cursor: 'pointer', fontSize: '0.9rem'
                }} onMouseOver={e => e.target.style.color = 'var(--color-text)'} onMouseOut={e => e.target.style.color = 'var(--color-muted)'}>
                  ← Trước
                </button>
              )}
              <button onClick={nextStep} style={{
                background: 'var(--color-accent)', color: 'white', border: 'none', borderRadius: '999px', padding: '6px 16px', fontWeight: 600, cursor: 'pointer', fontSize: '0.9rem',
                transition: 'transform 0.2s'
              }} onMouseOver={e => e.target.style.transform = 'translateY(-1px)'} onMouseOut={e => e.target.style.transform = 'translateY(0)'}>
                {currentStep === TOUR_STEPS.length ? 'Hoàn thành ✓' : 'Tiếp →'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default TourTooltip;
