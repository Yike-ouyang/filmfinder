import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { QuickTasteCTA } from "@/components/home/QuickTasteCTA";

describe("QuickTasteCTA", () => {
  it("invites visitors without a Letterboxd account", () => {
    render(<QuickTasteCTA />);

    expect(screen.getByText("No Letterboxd account?")).toBeVisible();
    expect(
      screen.getByText(/Rate 20 movies and we'll discover your taste/),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: /Discover My Taste/ }),
    ).toHaveAttribute("href", "/taste-profile");
  });
});
