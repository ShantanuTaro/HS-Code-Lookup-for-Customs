# Bundled fonts

Both are redistributed under the SIL Open Font License 1.1, whose full text sits beside the files
they cover. Self-hosted rather than loaded from Google Fonts so the pages make no third-party
request and render identically offline.

| Family        | Files                        | Upstream                          | Licence           |
| ------------- | ---------------------------- | --------------------------------- | ----------------- |
| Inter         | `Inter-latin*.woff2`         | https://github.com/rsms/inter     | OFL 1.1, see file |
| IBM Plex Mono | `IBMPlexMono-*-latin*.woff2` | https://github.com/IBM/plex       | OFL 1.1, see file |

Latin and latin-ext subsets only. Inter is a variable font, so one file per subset covers weights
400-600; IBM Plex Mono is static and ships a file per weight. `../fonts.css` declares the faces.

To refresh, take the woff2 URLs from the Google Fonts CSS API with a modern browser User-Agent,
download them, and rewrite the `src:` URLs in `fonts.css` to point at `/static/fonts/`.
