import type { Metadata } from "next";
import { DiscoverPageContent } from "@/components/discover/DiscoverPageContent";

export const metadata: Metadata = {
  title: "Discover",
  description:
    "Swipe through personalized film recommendations generated from your Letterboxd taste.",
};

export default function DiscoverPage() {
  return <DiscoverPageContent />;
}
