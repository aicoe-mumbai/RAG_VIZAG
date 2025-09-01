import React, { createContext, useState, useEffect } from 'react';

export const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Check for existing dark mode preference and update state
  useEffect(() => {
    const savedTheme = sessionStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark-mode');
    }
  }, []);

  // Toggle dark mode and update local storage
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    if (isDarkMode) {
      document.documentElement.classList.remove('dark-mode');
      sessionStorage.setItem('theme', 'light');
    } else {
      document.documentElement.classList.add('dark-mode');
      sessionStorage.setItem('theme', 'dark');
    }
  }; 

  return (
    <ThemeContext.Provider value={{ isDarkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
}

export default ThemeProvider;