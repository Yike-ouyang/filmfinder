import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeAll, vi } from "vitest";
import type { ImgHTMLAttributes, SyntheticEvent } from "react";

vi.mock("next/image", async () => {
  const React = await import("react");

  interface MockImageProps extends ImgHTMLAttributes<HTMLImageElement> {
    fill?: boolean;
    preload?: boolean;
    priority?: boolean;
    quality?: number;
    sizes?: string;
  }

  return {
    default: ({
      fill,
      preload,
      priority,
      quality,
      sizes,
      ...props
    }: MockImageProps) => {
      void fill;
      void preload;
      void priority;
      void quality;
      void sizes;
      return React.createElement("img", {
        ...props,
        onLoad: (event: SyntheticEvent<HTMLImageElement>) =>
          props.onLoad?.(event),
      });
    },
  };
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }),
  });

  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  class IntersectionObserverMock {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds = [];
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }

  vi.stubGlobal("ResizeObserver", ResizeObserverMock);
  vi.stubGlobal("IntersectionObserver", IntersectionObserverMock);
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  cleanup();
  window.sessionStorage.clear();
});
