# Medical AI Suite visual system

## Direction

Phase 9 uses an evidence-index composition: a concise research introduction
anchors the page beside a scannable list of the five persisted models. The
interface is designed for researchers and technical reviewers who need to
understand provenance and limitations before entering measurements.

## Materials and type

The surface is a warm off-white canvas with crisp white panels, subtle cool
green-gray rules, and soft tinted depth. Deep ink/navy is the reading color;
medical teal is reserved for actions, links, and research metadata. Parkinson's
Experimental status alone uses muted amber. There are no gradients, glass
effects, stock images, or decorative AI motifs.

The product uses a local system sans stack for stable rendering in the release
image. Headings are compact and weight-led; body copy stays within readable
line lengths. Measurement values and identifiers use tabular or wrapped
numeric presentation where useful.

## Component grammar

- Top navigation is a simple horizontal header that wraps at narrow widths.
- Model rows carry a number, name/dataset pair, release badge, and arrow affordance.
- Panels use one quiet border plus soft tinted depth and 8/12/16px radii.
- Buttons have 48px minimum targets, teal primary and white secondary variants.
- Focus-visible rings use a high-contrast teal outline; reduced motion removes
  decorative transitions and loading choreography.
- Long-form pages use a readable content column with a compact sticky/stacked
  wayfinding panel.
- Assessment forms retain the generated contract fields and shared wizard
  semantics, with denser field grouping and a calm research-use notice.
- Results expose the uncalibrated score and fixed threshold as text facts, not a
  probability or risk visualization.

## Responsive rules

The index becomes a single column below 860px. Navigation and model rows wrap
below 640px, with the status badge on its own line and no horizontal overflow.
Assessment fields and result facts collapse to one column on narrow screens.

## PDF companion

Reports use the same ink, teal, muted amber, warm-white, and line vocabulary.
Headers identify the suite and research report; metadata, intended use,
limitations, input measurements, and the prominent research-use disclaimer are
grouped in a stable page flow with a footer and page number. Content semantics
and in-memory generation remain unchanged.
