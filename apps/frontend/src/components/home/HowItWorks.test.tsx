import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { HowItWorks } from "@/components/home/HowItWorks";

describe("HowItWorks", () => {
  it("links to the Letterboxd export page and explains the ZIP upload", async () => {
    const user = userEvent.setup();
    render(<HowItWorks />);
    await user.click(
      screen.getByRole("button", { name: /How it works/ }),
    );
    const dialog = await screen.findByRole("dialog");
    const link = within(dialog).getByRole("link", {
      name: /Export My Letterboxd Data/,
    });
    expect(link).toHaveAttribute(
      "href",
      "https://letterboxd.com/user/exportdata/",
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(
      within(dialog).getByText(
        /Do not unzip it or upload individual CSV files/i,
      ),
    ).toBeVisible();
  });
});
