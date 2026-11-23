import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AboutSection } from "@/components/home/AboutSection";

describe("AboutSection", () => {
  it("shows the project credit and safe external links", () => {
    render(<AboutSection />);
    expect(
      screen.getByRole("heading", { name: "About FilmFinder" }),
    ).toBeVisible();
    expect(screen.getByText("Created by Yike Ouyang.")).toBeVisible();
    const github = screen.getByRole("link", { name: /GitHub/ });
    const linkedin = screen.getByRole("link", { name: /LinkedIn/ });
    expect(github).toHaveAttribute(
      "href",
      "https://github.com/Yike-ouyang",
    );
    expect(linkedin).toHaveAttribute(
      "href",
      "https://www.linkedin.com/in/yike-ouyang-8169312b7/",
    );
    expect(github).toHaveAttribute("target", "_blank");
    expect(github).toHaveAttribute("rel", "noopener noreferrer");
  });
});
