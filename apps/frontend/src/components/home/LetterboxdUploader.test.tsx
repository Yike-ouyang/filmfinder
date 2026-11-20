import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LetterboxdUploader } from "@/components/home/LetterboxdUploader";
import { ApiError, requestRecommendations } from "@/lib/api";
import { loadBatch } from "@/lib/storage";
import { makeResponse } from "@/test/fixtures";

const navigation = vi.hoisted(() => ({
  push: vi.fn(),
  prefetch: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    requestRecommendations: vi.fn(),
  };
});

const requestMock = vi.mocked(requestRecommendations);

function zipFile(name = "export.zip") {
  return new File(["zip-content"], name, { type: "application/zip" });
}

beforeEach(() => {
  navigation.push.mockReset();
  navigation.prefetch.mockReset();
  requestMock.mockReset();
});

describe("LetterboxdUploader", () => {
  it("selects a ZIP and starts the upload before navigating", async () => {
    const user = userEvent.setup();
    requestMock.mockResolvedValue(makeResponse());
    render(<LetterboxdUploader />);

    expect(navigation.prefetch).toHaveBeenCalledWith("/discover");

    const input = screen.getByLabelText("Letterboxd ZIP export");
    await user.upload(input, zipFile());

    expect(screen.getByText("export.zip")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Analyze my taste/ }));

    expect(await screen.findByText("Your discoveries are ready")).toBeVisible();
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(loadBatch().status).toBe("ready");

    await waitFor(
      () => expect(navigation.push).toHaveBeenCalledWith("/discover"),
      { timeout: 2500 },
    );
  });

  it("rejects files that are not ZIP archives", async () => {
    render(<LetterboxdUploader />);
    const input = screen.getByLabelText("Letterboxd ZIP export");

    fireEvent.change(input, {
      target: {
        files: [new File(["text"], "notes.txt", { type: "text/plain" })],
      },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Please choose a .zip file",
    );
    expect(requestMock).not.toHaveBeenCalled();
  });

  it("shows a friendly backend error and keeps the file selected", async () => {
    const user = userEvent.setup();
    requestMock.mockRejectedValue(new ApiError("no_matched_ratings", 422));
    render(<LetterboxdUploader />);

    await user.upload(
      screen.getByLabelText("Letterboxd ZIP export"),
      zipFile(),
    );
    await user.click(screen.getByRole("button", { name: /Analyze my taste/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "could not match enough",
    );
    expect(screen.getByText("export.zip")).toBeVisible();
    expect(navigation.push).not.toHaveBeenCalled();
  });

  it("shows a timeout error instead of treating it as a cancellation", async () => {
    const user = userEvent.setup();
    requestMock.mockRejectedValue(new ApiError("timeout", null));
    render(<LetterboxdUploader />);

    await user.upload(
      screen.getByLabelText("Letterboxd ZIP export"),
      zipFile(),
    );
    await user.click(screen.getByRole("button", { name: /Analyze my taste/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "took too long",
    );
  });

  it("lets the user remove the selected file", async () => {
    const user = userEvent.setup();
    render(<LetterboxdUploader />);

    await user.upload(
      screen.getByLabelText("Letterboxd ZIP export"),
      zipFile(),
    );
    await user.click(
      screen.getByRole("button", { name: "Remove selected file" }),
    );

    expect(screen.queryByText("export.zip")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Import Letterboxd/ }),
    ).toBeVisible();
  });
});
