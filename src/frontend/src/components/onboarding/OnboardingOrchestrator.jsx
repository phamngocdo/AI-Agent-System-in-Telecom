import React from 'react';
import { useOnboarding } from '../../context/OnboardingContext';
import WelcomeModal from './WelcomeModal';
import TourTooltip from './TourTooltip';

function OnboardingOrchestrator() {
  const { isActive, currentStep } = useOnboarding();

  if (!isActive) return null;

  if (currentStep === 0) {
    return <WelcomeModal />;
  }

  return <TourTooltip />;
}

export default OnboardingOrchestrator;
