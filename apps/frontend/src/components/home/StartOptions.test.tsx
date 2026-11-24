import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StartOptions } from "@/components/home/StartOptions";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

describe("StartOptions", () => {
  it("shows the import path and opens the no-account options in a dialog", async () => {
    const user = userEvent.setup();
    render(<StartOptions />);

    expect(screen.queryByText("Option 1")).toBeNull();
    expect(
      screen.getByRole("heading", { name: "Import Letterboxd" }),
    ).toBeVisible();
    expect(
      screen.getByText(
        /Already using Letterboxd\? Import your movie history/,
      ),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: /Import My Movies/ }),
    ).toBeVisible();

    const trigger = screen.getByRole("button", { name: /No account\?/ });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("heading", { name: "Browse Movie Library" }),
    ).toBeNull();

    await user.click(trigger);

    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("dialog")).toBeVisible();
    expect(
      screen.getByRole("heading", {
        name: "Choose how you'd like to indicate your preferences",
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Browse Movie Library" }),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: /Browse Movies/ }),
    ).toHaveAttribute("href", "/films");

    expect(
      screen.getByRole("heading", { name: "Quick Taste Profile" }),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: /Rate 20 Movies/ }),
    ).toHaveAttribute("href", "/taste-profile");
  });
});
