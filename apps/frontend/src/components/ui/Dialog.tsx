"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { X } from "lucide-react";
import {
  useEffect,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/cn";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  titleId: string;
  children: ReactNode;
  className?: string;
  showClose?: boolean;
  closeLabel?: string;
}

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

const subscribeToClient = () => () => undefined;
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

export function Dialog({
  open,
  onClose,
  titleId,
  children,
  className,
  showClose = true,
  closeLabel = "Close dialog",
}: DialogProps) {
  const mounted = useSyncExternalStore(
    subscribeToClient,
    getClientSnapshot,
    getServerSnapshot,
  );
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const previouslyFocused = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const focusTimer = window.setTimeout(() => {
      const focusable = panelRef.current?.querySelectorAll<HTMLElement>(
        FOCUSABLE_SELECTOR,
      );
      const first = focusable?.[0] ?? panelRef.current;
      first?.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) {
        return;
      }
      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      ).filter((element) => element.offsetParent !== null);
      if (focusable.length === 0) {
        event.preventDefault();
        panelRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [open]);

  if (!mounted) {
    return null;
  }

  return createPortal(
    <AnimatePresence>
      {open ? (
        <motion.div
          key="dialog-overlay"
          className="fixed inset-0 z-[80] flex items-end justify-center bg-black/75 p-0 backdrop-blur-sm sm:items-center sm:p-6"
          initial={reduceMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduceMotion ? 0 : 0.25 }}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              onClose();
            }
          }}
        >
          <motion.div
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            tabIndex={-1}
            className={cn(
              "safe-bottom relative max-h-[92svh] w-full max-w-lg overflow-y-auto rounded-t-3xl border border-white/10 bg-[#0d0d0d] shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] sm:rounded-3xl",
              className,
            )}
            initial={reduceMotion ? false : { opacity: 0, y: 32, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 24, scale: 0.98 }}
            transition={{
              duration: reduceMotion ? 0 : 0.32,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            {showClose ? (
              <button
                type="button"
                onClick={onClose}
                aria-label={closeLabel}
                className="absolute right-4 top-4 z-10 grid h-10 w-10 place-items-center rounded-full border border-white/10 bg-black/40 text-white/60 transition-colors hover:bg-white/10 hover:text-white"
              >
                <X aria-hidden="true" className="h-4 w-4" />
              </button>
            ) : null}
            {children}
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>,
    document.body,
  );
}
