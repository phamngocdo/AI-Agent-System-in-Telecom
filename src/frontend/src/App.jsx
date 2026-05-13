import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AppProvider, useAppContext } from './context/AppContext';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SettingsModal from './components/Modals/SettingsModal';
import ToastContainer from './components/ToastContainer';
import { OnboardingProvider } from './context/OnboardingContext';
import OnboardingOrchestrator from './components/onboarding/OnboardingOrchestrator';

import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';

function ProtectedRoute({ children }) {
  const { user, authLoading } = useAppContext();
  if (authLoading) {
    return null;
  }

  if (!user || !user.loggedIn) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function GuestRoute({ children }) {
  const { user, authLoading } = useAppContext();
  if (authLoading) {
    return null;
  }

  if (user && user.loggedIn) {
    return <Navigate to="/chat" replace />;
  }

  return children;
}

function MainLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(window.innerWidth <= 768);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeModal, setActiveModal] = useState(null);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 768) {
        setSidebarCollapsed(true);
      } else {
        setMobileOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const toggleSidebar = () => {
    if (window.innerWidth <= 768) {
      setMobileOpen(!mobileOpen);
    } else {
      setSidebarCollapsed(!sidebarCollapsed);
    }
  };

  const openModal = (modalName) => {
    setActiveModal(modalName);
  };

  const closeModal = () => {
    setActiveModal(null);
  };

  return (
    <div id="app">
      {/* Nền mờ cho mobile khi mở sidebar */}
      {mobileOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={() => setMobileOpen(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 15
          }}
        />
      )}
      <Sidebar 
        collapsed={sidebarCollapsed}
        mobileOpen={mobileOpen}
        toggleSidebar={toggleSidebar}
        openSettings={() => openModal('settings')} 
      />
      <main className="main" id="mainArea">
        <ChatArea />
      </main>

      <ToastContainer />
      <div className="ctx-menu" id="ctxMenu" style={{display: 'none'}}></div>

      {activeModal === 'settings' && <SettingsModal closeModal={closeModal} />}
      <OnboardingOrchestrator />
    </div>
  );
}


function PageTransitionWrapper({ children }) {
  const location = useLocation();
  return (
    <div key={location.pathname} style={{ animation: 'fadeIn 0.3s var(--ease-default) both', width: '100%', height: '100%' }}>
      {children}
    </div>
  );
}

function AppRoutes() {
  return (
    <PageTransitionWrapper>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<GuestRoute><Login /></GuestRoute>} />
        <Route path="/register" element={<GuestRoute><Register /></GuestRoute>} />
        <Route 
          path="/chat" 
          element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          } 
        />
        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </PageTransitionWrapper>
  );
}

function App() {
  return (
    <AppProvider>
      <OnboardingProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </OnboardingProvider>
    </AppProvider>
  );
}

export default App;
