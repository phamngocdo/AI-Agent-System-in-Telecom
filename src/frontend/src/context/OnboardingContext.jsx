import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAppContext } from './AppContext';

const OnboardingContext = createContext(null);

export function OnboardingProvider({ children }) {
  const { user } = useAppContext();
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const storageKey = user?.email ? `onboarding_completed:${user.email}` : null;

  useEffect(() => {
    if (!storageKey || !user?.loggedIn) {
      setIsActive(false);
      setCurrentStep(0);
      return;
    }

    const done = localStorage.getItem(storageKey);
    if (!done) {
      const t = setTimeout(() => setIsActive(true), 800);
      return () => clearTimeout(t);
    }
  }, [storageKey, user?.loggedIn]);

  const startTour = useCallback(() => {
    setIsActive(true);
    setCurrentStep(0);
  }, []);

  const skipTour = useCallback(() => {
    setIsActive(false);
    setCurrentStep(0);
    if (storageKey) {
      localStorage.setItem(storageKey, 'true');
    }
  }, [storageKey]);

  const nextStep = useCallback(() => {
    setCurrentStep(s => {
      if (s >= 4) {
        skipTour();
        return s;
      }
      return s + 1;
    });
  }, [skipTour]);

  const prevStep = useCallback(() => {
    setCurrentStep(s => Math.max(1, s - 1));
  }, []);

  return (
    <OnboardingContext.Provider value={{ isActive, currentStep, startTour, nextStep, prevStep, skipTour, storageKey }}>
      {children}
    </OnboardingContext.Provider>
  );
}

export const useOnboarding = () => {
  const ctx = useContext(OnboardingContext);
  if (!ctx) throw new Error('useOnboarding must be used inside OnboardingProvider');
  return ctx;
};
