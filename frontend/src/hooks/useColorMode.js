import { createContext, useContext } from 'react';

export const ColorModeContext = createContext(null);

export function useColorMode() {
  return useContext(ColorModeContext);
}
