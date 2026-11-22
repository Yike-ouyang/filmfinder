import { AboutSection } from "@/components/home/AboutSection";
import { HeroSection } from "@/components/home/HeroSection";

export default function HomePage() {
  return (
    <main id="main" className="relative">
      <HeroSection />
      <AboutSection />
    </main>
  );
}
