# MemoRizz for Mac

`MemoRizz.app` shows the local UI in its own window: it shows a UI already
running on port 8765, or runs `memorizz ui` and stops it on quit, and offers
to install MemoRizz with uv when it isn't installed. It is a native AppKit
and WebKit app in one Swift file, `main.swift`; `make_icon.swift` draws the
icon from the website's mark.

## Build

```bash
packaging/macos/build.sh              # dist/macos/MemoRizz.app and MemoRizz.dmg
```

It needs the Xcode command line tools (`xcode-select --install`) and builds a
universal binary (Apple silicon and Intel) for macOS 13 or later.

- `MACOS_SIGN_IDENTITY`: a Developer ID Application identity to sign with.
  Without one the app is signed ad hoc, and macOS blocks the first launch
  until the user chooses *Open Anyway* in System Settings → Privacy &
  Security.
- `MACOS_NOTARY_PROFILE`: a `xcrun notarytool store-credentials` profile; the
  .dmg is then notarized and stapled, and opens without that step.
- `MEMORIZZ_APP_VERSION`: the version shown in About (default: pyproject's).

## Publish

The website serves the .dmg from `public/downloads/MemoRizz.dmg` in the
`memorizz-website` repo, linked from the *Download for Mac* button
(`src/components/site/MacDownload.astro`). To update it, build, copy the
.dmg there, and push the website.

## Try it without touching your own setup

```bash
open -n --env HOME=/tmp/memorizz-home --env MEMORIZZ_UI_PORT=8799 \
  --env MEMORIZZ_APP_COMMAND="$(command -v memorizz)" dist/macos/MemoRizz.app
```

`MEMORIZZ_UI_PORT` picks the port and `MEMORIZZ_APP_COMMAND` the `memorizz`
to run; with `HOME` elsewhere, the server's store and the app's log
(`~/Library/Logs/MemoRizz/ui.log`) live there too.
