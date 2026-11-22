import type { Metadata } from "next";
import { TasteProfilePageContent } from "@/components/taste-profile/TasteProfilePageContent";

export const metadata: Metadata = {
  title: "Quick Taste Profile",
  description:
    "Rate 20 films and discover movie recommendations made for your taste, no Letterboxd account required.",
};

export default function TasteProfilePage() {
  return <TasteProfilePageContent />;
}
