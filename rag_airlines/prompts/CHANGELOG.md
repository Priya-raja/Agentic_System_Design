# Prompt changelog

## v6

- Distinguish real supplied legislation from fictional AeroNova policies.
- Preserve legal scope and effective dates; avoid individual eligibility conclusions
  when journey facts are missing.

## v2

- Added minimal-citation requirement.
- Added primary-source preference.
- Added explicit refusal for conflicting evidence.

## v1

- Initial grounded AeroNova answer prompt.
- Required citations and date-aware answers.

#v3

- its answered wrong about the menu in DXB to London.
so changed the prompt

## v5

- Explain and cite explicit source limitations (such as the unspecified Premium
  Economy allowance) instead of returning a generic missing-context refusal.
- Preserve sample-menu and route restrictions; never fill missing policy facts.
- Require a specific reason for genuinely missing or conflicting evidence.
- New prompt hash/version naturally invalidates older exact and semantic answers.
