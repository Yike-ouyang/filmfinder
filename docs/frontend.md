> Historical design brief. The current application also includes `/films`,
> account pages, My Lists and the manual profile flow. See
> `docs/film-library.md` and `apps/frontend/README.md`.

# FILMFINDER — Build a Cinematic Movie Discovery Website

## 1. Project Overview

You are an expert frontend developer, creative web designer, and UI/UX engineer.

Your mission is to build **FilmFinder**, a modern movie recommendation website powered by a machine learning algorithm.

FilmFinder uses a Factorization Machine (M5) trained on Letterboxd movie ratings to generate personalized movie recommendations.

The application should allow users to:

1. Import their Letterboxd profile.
2. Generate personalized movie recommendations.
3. Discover recommended movies through an interactive swipe interface.
4. Indicate whether they are interested in a movie, not interested, or have already watched it.

The application must consist of **exactly two pages**:

- `/` — Import Letterboxd Profile
- `/discover` — Discover Movies

The primary objective is to create a beautiful, cinematic, and immersive website.

**The visual design must be strongly inspired by the two provided SweetFox screenshots.**

The screenshots are the primary visual references for this project.

Do not build a generic streaming dashboard, a typical SaaS application, or an ordinary Tinder clone.

Instead, combine:

**SweetFox visual identity + Letterboxd profile import + interactive movie discovery.**

---

# 2. Visual References — IMPORTANT

Two screenshots of SweetFox have been provided.

Study them carefully before implementing the frontend.

These screenshots establish the target visual direction.

## Reference 1 — Movie Catalogue

The first screenshot shows a dark streaming interface with several horizontal rows of movies.

Important visual characteristics:

### Background

The background is extremely dark, almost black.

It is not pure black.

It has a subtle cinematic texture with very slight variations in brightness.

Approximate background colors:

```css
--background-primary: #080a08;
--background-secondary: #101310;
```

Avoid visible gradients covering entire sections unless they are derived from cinematic imagery.

The background should feel atmospheric and unobtrusive.

### Movie Cards

Movies are displayed as large horizontal thumbnails.

The thumbnails have:

- A 16:9 aspect ratio.
- Slightly rounded corners.
- High-quality cinematic imagery.
- No visible border.
- No unnecessary card background.
- No heavy shadow.
- Minimal spacing between the image and text.

Below each image:

1. Movie title in white.
2. Rating and metadata in muted gray.

Example:

```text
The Odyssey
★ 8.0 · 2026 · HD
```

The thumbnails are arranged in horizontally scrollable carousels.

Approximately five full cards are visible on a wide desktop screen.

Additional cards are partially visible at the edge, indicating horizontal scrolling.

### Section Headings

Headings are simple and compact.

Examples from the reference:

- New this week
- New in 4K
- New episodes

They are displayed in white with medium-to-semibold weight.

There are no excessive decorations or colored underlines.

The emphasis remains on the images.

### Spacing

Use generous spacing between horizontal sections.

Within each section, the movie thumbnails should be relatively close together.

Do not create excessive gaps between cards.

### General Impression

The interface should feel:

- Premium.
- Dark.
- Cinematic.
- Minimal.
- Image-focused.
- Smooth.
- Carefully aligned.

---

## Reference 2 — Cinematic Homepage

The second screenshot is the most important reference for FilmFinder's overall visual structure.

It shows a large cinematic hero section with a full-width background image.

### Hero Background

The background uses a high-resolution scene from a movie.

The image fills almost the entire screen.

It is not displayed inside a container.

It extends from edge to edge.

Use:

```css
background-size: cover;
background-position: center;
```

Apply dark gradients to improve readability.

For example:

```css
background:
  linear-gradient(
    to bottom,
    rgba(0, 0, 0, 0.15) 0%,
    rgba(0, 0, 0, 0.10) 35%,
    rgba(0, 0, 0, 0.75) 100%
  );
```

Adjust the gradient to the underlying image.

The image should remain clearly visible.

**Do not make it excessively dark or blurry.**

The goal is to preserve the cinematic atmosphere.

### Floating Navigation Bar

At the top center of the page, the reference features a floating navigation bar.

This is an essential design element.

Its characteristics:

- Horizontally centered.
- Positioned near the top of the viewport.
- Dark translucent gray background.
- Fully rounded pill shape.
- Compact dimensions.
- White text.
- White pill-shaped background for the active tab.
- Subtle separators.
- Minimal icons.

Approximate styling:

```css
.navigation {
  background: rgba(45, 50, 51, 0.82);
  backdrop-filter: blur(18px);
  border-radius: 999px;
  padding: 8px 12px;
}
```

The active navigation item should look like:

```css
.navigation-active {
  background: #ffffff;
  color: #111111;
  border-radius: 999px;
}
```

Adapt the navigation content for FilmFinder.

Suggested structure:

```text
[ FilmFinder ]  [ Import ]  [ Discover ]  |  [ ? ]
```

Use a small custom FilmFinder logo or typographic wordmark.

Do not reproduce the SweetFox branding.

### Hero Content

In the reference, a large movie title appears on the left side.

Below it are two rounded buttons.

The title appears directly over the background image.

There is no visible container surrounding it.

For FilmFinder, preserve this composition.

Use large white typography and simple rounded CTA buttons.

### Bottom Carousel

At the bottom of the hero section, the reference shows a horizontal row of movie thumbnails.

This row is partially integrated into the hero background.

The thumbnails have a 16:9 ratio.

Each contains a cinematic still.

The row overlaps visually with the darker portion of the hero.

**This arrangement is an important part of the desired design.**

Use it as inspiration for FilmFinder's cinematic previews.

---

# 3. Global Design Direction

The final website should closely follow the visual language of these references.

## 3.1. Design Philosophy

Prioritize:

- Full-width cinematic backgrounds.
- High-quality movie imagery.
- Dark backgrounds.
- White typography.
- Compact rounded navigation.
- Minimal overlays.
- Soft gradients.
- Horizontal movie carousels.
- Clean alignment.
- Smooth transitions.

Avoid:

- Brightly colored backgrounds.
- Purple or pink gradients.
- Generic SaaS components.
- Dashboard-style interfaces.
- Excessive glassmorphism.
- Large decorative borders.
- Neumorphism.
- Overly animated elements.
- Too many UI controls.
- Excessive text.

FilmFinder must feel like a premium streaming service.

However, unlike a conventional streaming service, its primary purpose is movie discovery through machine learning.

---

# 4. Color Palette

Use the following approximate design tokens, inspired by the screenshots.

```css
:root {
  --background-primary: #080a08;
  --background-secondary: #111411;

  --text-primary: #ffffff;
  --text-secondary: #a1a1a1;
  --text-muted: #777777;

  --navbar-background: rgba(45, 50, 51, 0.82);

  --button-primary: #ffffff;
  --button-primary-text: #111111;

  --button-secondary: rgba(255, 255, 255, 0.20);

  --border-subtle: rgba(255, 255, 255, 0.12);

  --radius-card: 12px;
  --radius-button: 999px;
}
```

These values are implementation suggestions, not exact colors extracted from SweetFox.

Fine-tune them by comparing the rendered frontend with the reference screenshots.

Use accent colors very sparingly.

The posters and movie scenes should provide most of the visual color.

---

# 5. Typography

Use a modern, neutral sans-serif typeface.

Suggested options:

- Inter.
- Manrope.
- Geist.

Prefer a font that closely matches the reference screenshots.

### Typography Hierarchy

Hero heading:

- Large.
- Bold.
- White.
- Tight letter spacing.
- Clean line breaks.

Section headings:

- Medium size.
- Semibold.
- White.

Movie titles:

- Medium weight.
- White.
- Compact line height.

Metadata:

- Small.
- Muted gray.

Avoid decorative fonts.

The visual identity should come primarily from cinematography and composition, not unusual typography.

---

# 6. Technology Stack

Implement the frontend using:

### Framework

**Next.js with App Router**

### Language

**TypeScript**

### Styling

**Tailwind CSS**

### Animation

**Motion for React**

### Icons

**Lucide React**

### State Management

Use React state and hooks initially.

Introduce Zustand only if shared application state becomes difficult to manage.

### Backend

Prepare an API integration layer for:

**FastAPI + Python + FilmFinder M5 recommendation model**

Use a clean separation between frontend components and API communication.

### Data Handling

Use typed interfaces for:

- Movie.
- User profile.
- Import result.
- Recommendation.
- Swipe feedback.

Do not introduce unnecessary dependencies.

---

# 7. PAGE ONE — LETTERBOXD IMPORT

## Main Objective

Create a cinematic landing page where the user can import their Letterboxd profile.

This page should be visually close to the second SweetFox screenshot.

The experience should feel like entering a premium streaming platform.

## 7.1. Overall Layout

The page should contain:

1. Floating navigation bar.
2. Full-width cinematic hero.
3. Large headline on the left.
4. Letterboxd import interaction.
5. Horizontal movie thumbnail carousel at the bottom.

The hero should occupy most of the initial viewport.

The page should feel immersive immediately upon loading.

---

## 7.2. Hero Background

Use a large cinematic movie still as the background.

The image should:

- Cover the entire viewport width.
- Have a wide cinematic composition.
- Contain visually interesting scenery.
- Have enough negative space on the left for text.
- Remain visible behind the interface.

Prefer a landscape image rather than a movie poster.

Apply a dark gradient on the left and bottom.

The background should resemble the large movie scene used on SweetFox's homepage.

Avoid using a generic abstract gradient instead of a movie scene.

### Background Animation

Optional:

Add a very subtle background scale animation.

For example, a slow zoom from 1.0 to 1.03.

The animation should be barely noticeable.

Respect reduced-motion preferences.

---

## 7.3. Floating Navigation

Place the floating navigation at the top center.

Suggested content:

```text
FILMFINDER    Import    Discover    ?
```

The Import tab should be active.

Use the same pill-shaped navigation style as the SweetFox reference.

The navigation should remain readable over the background image.

---

## 7.4. Hero Text

Position the main content toward the left side of the hero.

Suggested headline:

**YOUR NEXT FAVORITE MOVIE.**

Below:

"Discover movies you'll love, powered by your Letterboxd taste."

Keep the text short.

The main title should be large, bold, and white.

Do not put the headline inside a large opaque card.

It should appear directly over the cinematic background.

---

## 7.5. Upload Interaction

Below the headline, create a prominent upload interaction.

Use a simple, elegant design inspired by the buttons in the SweetFox screenshot.

Primary CTA:

**Import Letterboxd Profile**

Suggested appearance:

- White background.
- Black text.
- Fully rounded shape.
- Small upload icon.
- Smooth hover effect.

Secondary action:

**How to export my data?**

Use a translucent gray pill button.

### Upload Behavior

Clicking the primary button should open a native file picker.

Accept:

- Letterboxd ZIP exports.
- Letterboxd ratings CSV files.

Also support drag-and-drop.

When a file is dragged over the page, display a subtle centered drop-zone overlay.

Avoid displaying a huge dashed rectangle permanently in the middle of the hero.

The upload interface should remain visually minimal.

### Selected File

Once a file is selected, display a compact floating status panel:

```text
ratings.csv
File ready to import

[Import profile]
```

Use the same dark translucent styling as the navigation.

---

# 8. Import Loading State

After the user confirms the upload, display a polished loading experience.

Keep the cinematic background visible.

Use a small centered panel or overlay.

Suggested messages:

```text
Reading your movie history...
Matching your movies...
Preparing your recommendations...
```

Use subtle animations.

Avoid fake progress percentages.

When the backend provides actual progress information, display it.

Do not simulate successful personalization without real data.

---

# 9. Import Success State

When the profile is ready, update the hero content.

Suggested presentation:

```text
YOUR PROFILE IS READY.

247 movies imported.
231 movies matched.

Your next favorite movie is waiting.
```

Use actual backend values.

Display a large white pill-shaped button:

**Start discovering →**

Clicking the button navigates to `/discover`.

Use a smooth fade transition.

Do not trigger unnecessary full-page reloads.

---

# 10. Bottom Cinematic Carousel

The bottom part of the import page should reproduce the visual style of the SweetFox "Top 10 Today" row.

Create a horizontal carousel of movie stills.

Suggested section title:

**A WORLD OF CINEMA**

Display approximately 5–6 landscape thumbnails on desktop.

Each thumbnail should have:

- 16:9 aspect ratio.
- Rounded corners.
- High-quality still.
- Movie title underneath.
- Release year and genres in muted text.

The carousel should be integrated into the hero background.

Avoid heavy separation between the hero and the carousel.

### Interaction

Support:

- Horizontal scrolling.
- Arrow navigation on desktop.
- Touch scrolling on mobile.
- Subtle hover scale effects.

These movies are visual examples, not personalized recommendations before a profile has been imported.

---

# 11. PAGE TWO — DISCOVER

## Main Objective

Create an immersive movie discovery interface based on swipe gestures.

The user receives personalized movies and can indicate whether they are interested.

The interaction should resemble Tinder.

However, the visual design must remain consistent with SweetFox.

This is the most important page of FilmFinder.

---

# 12. Discover Page Layout

Use the same visual principles as the SweetFox homepage:

- Full-width cinematic background.
- Floating navigation.
- Dark gradient overlays.
- Large movie imagery.
- Minimal controls.
- Rounded buttons.
- Small metadata labels.

### Main Difference

Instead of displaying a large movie title with Play and Info buttons, the center of the experience should feature an interactive movie swipe card.

The main card uses the movie's **vertical theatrical poster**.

This is an intentional adaptation of the reference.

SweetFox uses landscape images for its catalogue thumbnails, while FilmFinder uses a vertical poster for the actual swipe interaction.

Use horizontal movie stills for backgrounds and preview carousels, and vertical posters for swipe cards.

Do not force all images into the same aspect ratio.

---

# 13. Cinematic Background

The Discover page should display a large cinematic background related to the currently active movie.

Use the movie's backdrop image, not its vertical poster, whenever available.

The image must cover the full available width.

Apply a dark gradient to ensure the foreground card remains readable.

Suggested composition:

- Full-width movie backdrop.
- Darker region behind the central card.
- Stronger gradient toward the bottom.
- Subtle transition when the active movie changes.

### Background Transition

When moving to the next recommendation:

1. Keep the current background visible during the card exit animation.
2. Load the next movie's backdrop.
3. Crossfade between the two images.
4. Preserve a dark overlay for readability.

Avoid abrupt changes in background imagery.

Use the poster as a fallback if no backdrop exists.

---

# 14. Swipe Card Design

The swipe card should be centered in the main content area.

Use a vertical **2:3 movie poster**.

Suggested width:

- Desktop: approximately 330–400 px.
- Mobile: responsive, with a maximum width adapted to the viewport.

The card should have:

- High-quality poster.
- Rounded corners.
- Soft shadow.
- No thick border.
- Subtle gradient near the bottom.
- Movie information integrated into the poster.

### Movie Information

At the bottom of the card, display:

**Movie Title**

Below:

`2024 · Drama · Thriller`

Optionally:

`Predicted rating: 9.2/10`

Only display predicted ratings when available.

Use readable white typography.

Apply a subtle dark gradient behind the title.

### Poster Stack

Show two additional cards behind the main card.

They should be slightly offset and scaled down.

This creates depth and communicates that more recommendations are waiting.

The stack should not distract from the active poster.

---

# 15. Swipe Interactions

Implement fully functional swipe gestures.

## Swipe Right

Meaning:

**Interested**

When dragging right:

- The card follows the pointer.
- It rotates slightly clockwise.
- A subtle green-tinted indicator appears.
- The text "INTERESTED" becomes visible.

After crossing the threshold, the card exits to the right.

Record the decision.

Move to the next recommendation.

## Swipe Left

Meaning:

**Not interested**

When dragging left:

- The card follows the pointer.
- It rotates slightly counterclockwise.
- A subtle red-tinted indicator appears.
- The text "PASS" becomes visible.

After crossing the threshold, the card exits to the left.

Record the decision.

Move to the next recommendation.

### Gesture Behavior

Use spring animations.

If the drag is insufficient, return the card smoothly to the center.

Prevent double submissions.

Avoid jerky movements.

The card should feel physically responsive.

---

# 16. Swipe Buttons

Below the card, display two large round buttons.

Left:

**X — Pass**

Right:

**Heart — Interested**

Buttons should resemble the minimal rounded controls used on SweetFox.

Use a neutral translucent background when inactive.

Use color only as interaction feedback.

Do not permanently display large green and red circles that dominate the interface.

Also include smaller secondary controls:

- Undo.
- Details.
- Already seen.

These controls should be visually subordinate to the two primary decisions.

All actions must work using mouse, touch, and keyboard.

---

# 17. Horizontal Movie Preview Carousel

To reinforce the visual connection with SweetFox, add a small horizontal movie preview section near the bottom of the Discover page.

Suggested heading:

**COMING UP**

Display the next movies in the recommendation batch.

Use the same thumbnail style as SweetFox:

- Landscape 16:9 imagery.
- Rounded corners.
- Title below.
- Gray metadata.
- Horizontal scrolling.

This section should be secondary to the swipe card.

### Important UX Constraint

Do not let the preview carousel compete with the main swipe interface.

On desktop, it can appear at the bottom of the hero or immediately below the primary swipe experience.

On mobile, prioritize the swipe card and controls.

The preview carousel can appear lower on the page.

---

# 18. Movie Details Overlay

Provide a Details button.

Clicking it should open a modal or overlay.

Do not navigate to a third page.

The overlay should contain:

- Large movie backdrop.
- Poster.
- Movie title.
- Release year.
- Genres.
- Runtime, if available.
- Synopsis.
- Predicted rating, if available.

Maintain the same dark cinematic design.

Use a translucent dark background.

The movie image should remain prominent.

Avoid turning the modal into a dense information dashboard.

---

# 19. Recommendation Progress

Display a discreet recommendation counter near the top of the Discover page.

Example:

```text
DISCOVERING  07 / 20
```

Use small white or muted-gray typography.

A thin progress indicator may be added.

Keep it understated.

The progress should update after every swipe.

### Completion State

After all 20 movies have been processed, display a cinematic completion screen within `/discover`.

Suggested headline:

**THAT'S A WRAP.**

Subheading:

"You've explored your latest recommendations."

Display:

- Number of movies marked Interested.
- Number of movies passed.
- Number of movies marked Already Seen.

Provide a button:

**Discover more movies**

If no additional recommendations are available, display an appropriate empty state rather than repeating the same movies automatically.

---

# 20. Responsive Design

The website must be responsive.

## Desktop

Prioritize:

- Large cinematic backgrounds.
- Spacious hero layouts.
- Centered floating navigation.
- Large movie thumbnails.
- Smooth swipe interactions.

The overall composition should resemble the desktop SweetFox screenshots.

## Mobile

Adapt the layout to a smartphone.

Requirements:

- Compact floating navigation.
- Readable hero text.
- Prominent upload CTA.
- Swipe card sized for the available viewport.
- Large touch-friendly buttons.
- Smooth touch gestures.
- No horizontal overflow.
- Safe-area support.

On mobile, the Discover page should feel similar to a native application.

The user must be able to swipe comfortably with one hand.

Do not let the navigation or preview carousel take too much space.

---

# 21. Backend Integration

The frontend will eventually connect to the FilmFinder recommendation engine.

The backend is implemented in Python using FastAPI.

The recommendation model is a contextual Factorization Machine called M5.

Create an abstraction layer for the following logical operations.

### Import Profile

```http
POST /profiles/import
```

Accept a Letterboxd ZIP or ratings CSV.

Return:

- Session identifier.
- Number of imported ratings.
- Number of matched movies.
- Import status.

### Retrieve Recommendations

```http
GET /recommendations?limit=20
```

Return:

- Movie ID.
- Title.
- Year.
- Genres.
- Poster URL.
- Backdrop URL.
- Predicted rating, if available.
- Description.

Use an appropriately scoped session mechanism.

### Record Feedback

```http
POST /feedback
```

Supported decisions:

```typescript
type SwipeDecision =
  | "interested"
  | "not_interested"
  | "already_seen";
```

Do not interpret a swipe as a numeric movie rating.

Do not automatically retrain M5 from swipe feedback.

The backend integration should follow the actual API contracts when they exist.

These routes are suggestions, not assumptions about the existing project.

---

# 22. Demo Mode

If the backend is not ready, implement a fully functional demo mode.

Use realistic movie data.

The demo must support:

- Simulated profile import.
- Navigation between pages.
- Displaying movie recommendations.
- Swiping left and right.
- Undo.
- Movie details.
- Tracking decisions.
- Recommendation completion.

Clearly indicate when the application uses demo recommendations.

Do not claim that mock scores come from M5.

Use real movie imagery only when it is available through an authorized source.

Otherwise, provide visually consistent placeholders.

The frontend architecture must allow replacing demo data with real API responses without redesigning the interface.

---

# 23. Animation Guidelines

Animations should be subtle and polished.

Use Motion for React.

### Navigation

- Smooth hover effects.
- Soft active-tab transition.
- No excessive movement.

### Hero

- Optional subtle entrance animation.
- Gentle fade-in.
- No large bouncing effects.

### Carousels

- Smooth horizontal scrolling.
- Subtle card scaling on hover.
- Arrow controls.

### Swipe Cards

- Natural dragging.
- Slight rotation.
- Spring return.
- Smooth exit.
- Seamless next-card entrance.

### Backgrounds

- Smooth crossfade between movie backdrops.

### Buttons

- Small scale feedback.
- Fast transitions.

Avoid excessive animation.

The website should feel premium, not playful to the point of becoming distracting.

---

# 24. Implementation Details

Suggested project structure:

```text
filmfinder/
│
├── src/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── discover/
│   │   │   └── page.tsx
│   │   ├── layout.tsx
│   │   └── globals.css
│   │
│   ├── components/
│   │   ├── navigation/
│   │   │   └── FloatingNavbar.tsx
│   │   │
│   │   ├── hero/
│   │   │   └── CinematicHero.tsx
│   │   │
│   │   ├── movies/
│   │   │   ├── MovieThumbnail.tsx
│   │   │   ├── MovieCarousel.tsx
│   │   │   └── MovieDetails.tsx
│   │   │
│   │   ├── upload/
│   │   │   ├── LetterboxdUpload.tsx
│   │   │   └── ImportStatus.tsx
│   │   │
│   │   └── discover/
│   │       ├── SwipeCard.tsx
│   │       ├── SwipeDeck.tsx
│   │       ├── SwipeControls.tsx
│   │       └── DiscoveryProgress.tsx
│   │
│   ├── hooks/
│   │   ├── useSwipe.ts
│   │   └── useRecommendations.ts
│   │
│   ├── lib/
│   │   └── api.ts
│   │
│   ├── data/
│   │   └── mockMovies.ts
│   │
│   └── types/
│       └── movie.ts
│
└── public/
    └── images/
```

Use reusable components.

Do not duplicate UI logic.

Keep the implementation maintainable.

---

# 25. Visual Quality Checklist

Before finishing, compare the generated frontend with the two provided SweetFox screenshots.

Check the following elements carefully.

### Screenshot 1 — Catalogue

Does FilmFinder reproduce:

- The almost-black background?
- The clean horizontal movie thumbnails?
- The 16:9 thumbnail proportions?
- The rounded corners?
- The white movie titles?
- The small gray metadata?
- The compact section headings?
- The horizontal carousel behavior?
- The restrained spacing?

### Screenshot 2 — Homepage

Does FilmFinder reproduce:

- The full-width cinematic hero?
- The large photographic background?
- The dark gradient overlays?
- The centered floating pill navigation?
- The white active navigation tab?
- The large white headline?
- The rounded primary buttons?
- The cinematic movie carousel at the bottom?

### Discover Page

Does the swipe interface feel visually integrated into the same design system?

The Discover page must not look like a separate application.

It should feel like an interactive feature built inside the same premium streaming interface.

---

# 26. Implementation and Verification Workflow

Follow these steps.

### Step 1 — Analyze the Screenshots

Before coding, identify the key visual rules from the supplied SweetFox screenshots.

Do not substitute generic design assumptions for what is visibly present in the references.

### Step 2 — Build the Design System

Implement:

- Colors.
- Typography.
- Spacing.
- Navigation.
- Movie thumbnails.
- Buttons.
- Background treatments.

### Step 3 — Build the Import Page

Implement the complete cinematic hero and Letterboxd upload interaction.

### Step 4 — Build the Discover Page

Implement the cinematic background, swipe card stack, controls, movie details, and progress indicator.

### Step 5 — Implement Functional Interactions

Ensure:

- Upload selection works.
- Demo mode works.
- Navigation works.
- Swipe gestures work.
- Feedback is stored.
- Undo works.
- Movie details open.
- Recommendation progress updates.

### Step 6 — Test Responsive Layouts

Test at least:

- Desktop: 1440px.
- Tablet: 768px.
- Mobile: 390px.
- Small mobile: 320px.

### Step 7 — Visual Verification

Run the application in a browser.

Capture screenshots of both pages.

Compare the result with the supplied SweetFox screenshots.

Correct visible problems in:

- Image proportions.
- Navigation positioning.
- Spacing.
- Font sizes.
- Background gradients.
- Card dimensions.
- Responsiveness.

Repeat until the result is visually consistent.

### Step 8 — Functional Verification

Verify the complete user journey:

```text
Open FilmFinder
        ↓
Import Letterboxd profile
        ↓
Profile successfully processed
        ↓
Open Discover page
        ↓
View first recommended movie
        ↓
Swipe right or left
        ↓
Save feedback
        ↓
Display next recommendation
        ↓
Complete recommendation batch
```

Check that there are no broken interactions or obvious runtime errors.

---

# 27. Expected Deliverables

Deliver:

1. A complete Next.js frontend.
2. A cinematic Letterboxd import page.
3. An interactive movie discovery page.
4. A floating SweetFox-inspired navigation bar.
5. Full-width cinematic backgrounds.
6. Horizontal movie carousels.
7. Functional swipe gestures.
8. Movie details overlays.
9. Responsive layouts.
10. Demo data mode.
11. A backend API integration layer.
12. Installation and execution instructions.

The application must be functional and visually polished.

Do not deliver only static mockups.

Do not stop at generating individual components.

Implement and verify the complete two-page experience.

---

# 28. Final Design Priority

**Visual fidelity to the supplied SweetFox screenshots is a major priority.**

Do not reinterpret the reference as a generic dark-themed web application.

The important characteristics are:

- The large cinematic backgrounds.
- The floating pill-shaped navigation.
- The almost-black interface.
- The landscape movie thumbnails.
- The clean white typography.
- The subtle gray metadata.
- The restrained use of color.
- The premium streaming-platform atmosphere.

The only major interaction difference should be the movie discovery experience.

FilmFinder replaces traditional streaming content consumption with an interactive personalized recommendation system.

The final result should feel like:

**SweetFox's cinematic interface transformed into a personalized movie discovery experience powered by FilmFinder.**