import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RatingStars } from "@/components/films/RatingStars";

describe("RatingStars", () => {
  it("renders ten half-star controls and reports the selected value", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<RatingStars value={null} onChange={onChange} label="Your rating" />);

    expect(screen.getAllByRole("radio")).toHaveLength(10);
    await user.click(
      screen.getByRole("radio", {
        name: "Rate 4.5 stars out of 5 (9/10)",
      }),
    );
    expect(onChange).toHaveBeenCalledWith(9);
  });

  it("does not render controls in read-only mode", () => {
    render(<RatingStars value={7} readOnly label="Average rating" />);
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
});
