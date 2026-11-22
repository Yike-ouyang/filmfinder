import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BuildProfileCTA } from "@/components/home/BuildProfileCTA";

describe("BuildProfileCTA", () => {
  it("links anonymous visitors to the film library", () => {
    render(<BuildProfileCTA />);
    expect(screen.getByText("No Letterboxd account?")).toBeVisible();
    expect(
      screen.getByText(/rate the movies you've watched/i),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: /Build My Movie Profile/ }),
    ).toHaveAttribute("href", "/films");
  });
});
