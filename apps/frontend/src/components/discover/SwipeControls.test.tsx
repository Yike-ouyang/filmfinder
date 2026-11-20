import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SwipeControls } from "@/components/discover/SwipeControls";

function setup(canUndo = true) {
  const handlers = {
    onUndo: vi.fn(),
    onPass: vi.fn(),
    onSeen: vi.fn(),
    onInterested: vi.fn(),
  };
  render(<SwipeControls canUndo={canUndo} {...handlers} />);
  return handlers;
}

describe("SwipeControls", () => {
  it("exposes all four actions and calls their handlers", async () => {
    const user = userEvent.setup();
    const handlers = setup();

    expect(screen.getByRole("group", { name: "Movie actions" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Undo/ }));
    await user.click(screen.getByRole("button", { name: /Pass/ }));
    await user.click(screen.getByRole("button", { name: /Already seen/i }));
    await user.click(screen.getByRole("button", { name: /Interested/ }));

    expect(handlers.onUndo).toHaveBeenCalledTimes(1);
    expect(handlers.onPass).toHaveBeenCalledTimes(1);
    expect(handlers.onSeen).toHaveBeenCalledTimes(1);
    expect(handlers.onInterested).toHaveBeenCalledTimes(1);
  });

  it("disables undo when there is no history", () => {
    setup(false);
    expect(screen.getByRole("button", { name: /Undo/ })).toBeDisabled();
  });
});
