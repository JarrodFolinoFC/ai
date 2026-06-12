import { createContext, useContext } from 'react';

// Current training step, provided by BigramFlow so any live-updating Panel can
// show a "live" badge with the step without threading the prop through every card.
export const StepContext = createContext<number | null>(null);

export const useStep = () => useContext(StepContext);
