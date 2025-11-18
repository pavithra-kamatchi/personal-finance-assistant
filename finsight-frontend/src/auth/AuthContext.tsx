// Authentication Context Provider
// Manages user authentication state and token across the app

import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { User, AuthContextType } from '../types';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setTokenState] = useState<string | null>(null);

  // Initialize token from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('access_token');
    if (storedToken) {
      setTokenState(storedToken);
      // Note: In a production app, you might want to validate the token
      // by calling an auth-check endpoint here
    }
  }, []);

  const setToken = (newToken: string | null) => {
    if (newToken) {
      localStorage.setItem('access_token', newToken);
      setTokenState(newToken);
    } else {
      localStorage.removeItem('access_token');
      setTokenState(null);
    }
  };

  const signOut = () => {
    localStorage.removeItem('access_token');
    setTokenState(null);
    setUser(null);
    window.location.href = '/';
  };

  const value: AuthContextType = {
    user,
    token,
    setUser,
    setToken,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
