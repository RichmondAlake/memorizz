#!/bin/bash
# Build MemoRizz.app (a universal binary) and MemoRizz.dmg.
#
#   packaging/macos/build.sh [output dir]      # default: dist/macos
#
# MACOS_SIGN_IDENTITY names a Developer ID Application identity to sign with
# (default: ad hoc, which macOS asks the user to allow on first launch), and
# MACOS_NOTARY_PROFILE a `notarytool store-credentials` profile to notarize
# and staple the .dmg with. The version is pyproject's unless
# MEMORIZZ_APP_VERSION says otherwise.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$ROOT/dist/macos}"
VERSION="${MEMORIZZ_APP_VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)}"
APP="${OUT:?}/MemoRizz.app"
WORK="${OUT:?}/build"

rm -rf "${APP:?}" "${WORK:?}" "${OUT:?}/MemoRizz.dmg"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$WORK"

for arch in arm64 x86_64; do
  xcrun swiftc -O -swift-version 5 -target "$arch-apple-macos13.0" \
    -framework AppKit -framework WebKit \
    -o "$WORK/MemoRizz-$arch" "$HERE/main.swift"
done
lipo -create -output "$APP/Contents/MacOS/MemoRizz" "$WORK/MemoRizz-arm64" "$WORK/MemoRizz-x86_64"
sed "s/__VERSION__/$VERSION/g" "$HERE/Info.plist" > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist" >/dev/null

xcrun swiftc -O -swift-version 5 -framework AppKit -o "$WORK/make_icon" "$HERE/make_icon.swift"
"$WORK/make_icon" "$WORK/AppIcon.iconset"
iconutil -c icns -o "$APP/Contents/Resources/AppIcon.icns" "$WORK/AppIcon.iconset"

IDENTITY="${MACOS_SIGN_IDENTITY:--}"
if [ "$IDENTITY" = "-" ]; then
  codesign --force --sign - --options runtime "$APP"
else
  codesign --force --sign "$IDENTITY" --options runtime --timestamp "$APP"
fi
codesign --verify --strict --verbose=1 "$APP"

STAGE="$WORK/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "MemoRizz" -srcfolder "$STAGE" -ov -format UDZO \
  "$OUT/MemoRizz.dmg"
if [ "$IDENTITY" != "-" ]; then
  codesign --force --sign "$IDENTITY" --timestamp "$OUT/MemoRizz.dmg"
fi
if [ -n "${MACOS_NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$OUT/MemoRizz.dmg" --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$OUT/MemoRizz.dmg"
fi
rm -rf "${WORK:?}"
echo "MemoRizz $VERSION: $OUT/MemoRizz.dmg"
