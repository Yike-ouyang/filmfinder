import { Briefcase, Code2, ExternalLink } from "lucide-react";

const links = [
  {
    href: "https://github.com/Yike-ouyang",
    label: "GitHub",
    icon: Code2,
  },
  {
    href: "https://www.linkedin.com/in/yike-ouyang-8169312b7/",
    label: "LinkedIn",
    icon: Briefcase,
  },
] as const;

export function AboutSection() {
  return (
    <section
      aria-labelledby="about-title"
      className="relative border-t border-white/10 bg-[#080808] px-5 py-16 sm:px-8"
    >
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-[10px] font-medium uppercase tracking-[0.32em] text-white/40">
          About
        </p>
        <h2
          id="about-title"
          className="mt-4 text-2xl font-semibold tracking-[-0.03em] text-white sm:text-3xl"
        >
          About FilmFinder
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-balance text-sm leading-relaxed text-white/50">
          FilmFinder is a personal movie recommendation project powered by
          machine learning, designed to help you discover films that match
          your taste.
        </p>
        <p className="mt-3 text-sm text-white/65">
          Created by Yike Ouyang.
        </p>
        <div className="mt-6 flex items-center justify-center gap-3">
          {links.map((link) => {
            const Icon = link.icon;
            return (
              <a
                key={link.href}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex h-10 items-center gap-2 rounded-full border border-white/10 px-4 text-xs font-medium text-white/60 transition-colors hover:border-white/25 hover:bg-white/5 hover:text-white"
              >
                <Icon aria-hidden="true" className="h-3.5 w-3.5" />
                {link.label}
                <ExternalLink
                  aria-hidden="true"
                  className="h-3 w-3 text-white/30"
                />
              </a>
            );
          })}
        </div>
      </div>
    </section>
  );
}
