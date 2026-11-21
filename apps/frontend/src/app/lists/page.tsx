import type { Metadata } from "next";
import { ListsPageContent } from "@/components/lists/ListsPageContent";

export const metadata: Metadata = {
  title: "My Lists",
  description:
    "Review the films you are interested in and the ones you want to skip.",
};

export default function ListsPage() {
  return <ListsPageContent />;
}
