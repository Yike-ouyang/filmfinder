import type { Metadata } from "next";
import { AdminPageContent } from "@/components/admin/AdminPageContent";

export const metadata: Metadata = {
  title: "Admin",
  description: "Private FilmFinder administration dashboard.",
  robots: {
    index: false,
    follow: false,
  },
};

export default function AdminPage() {
  return <AdminPageContent />;
}
