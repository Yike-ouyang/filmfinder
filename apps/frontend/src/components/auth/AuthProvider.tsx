"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  deleteAccount,
  fetchCurrentUser,
  login,
  logout,
} from "@/lib/auth";
import type { AuthUser } from "@/types/auth";

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<AuthUser>;
  signOut: () => Promise<void>;
  removeAccount: (password: string) => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const generation = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const current = await fetchCurrentUser();
      setUser(current);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    const currentGeneration = generation.current;
    fetchCurrentUser()
      .then((current) => {
        if (active && generation.current === currentGeneration) {
          setUser(current);
        }
      })
      .catch(() => {
        if (active && generation.current === currentGeneration) {
          setUser(null);
        }
      })
      .finally(() => {
        if (active && generation.current === currentGeneration) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const current = await login(email, password);
    generation.current += 1;
    setUser(current);
    setLoading(false);
    return current;
  }, []);

  const signOut = useCallback(async () => {
    try {
      await logout();
    } finally {
      generation.current += 1;
      setUser(null);
      setLoading(false);
    }
  }, []);

  const removeAccount = useCallback(async (password: string) => {
    await deleteAccount(password);
    generation.current += 1;
    setUser(null);
    setLoading(false);
  }, []);

  const value = useMemo(
    () => ({ user, loading, signIn, signOut, removeAccount, refresh }),
    [user, loading, signIn, signOut, removeAccount, refresh],
  );

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider.");
  }
  return context;
}
