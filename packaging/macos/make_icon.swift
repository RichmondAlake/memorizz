// Draws the MemoRizz mark (the site's favicon: an "M" traced through three
// memory nodes) as a macOS app icon, at every size an .iconset needs.
//
//   make_icon <output.iconset>

import AppKit

let output = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "AppIcon.iconset")
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> NSColor {
    NSColor(
        srgbRed: CGFloat((hex >> 16) & 0xff) / 255, green: CGFloat((hex >> 8) & 0xff) / 255,
        blue: CGFloat(hex & 0xff) / 255, alpha: alpha)
}

func draw(size: Int) -> Data {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(size) / 1024

    // The body, on the macOS icon grid: 824 points inside 1024, with a shadow.
    let body = NSRect(x: 100 * s, y: 100 * s, width: 824 * s, height: 824 * s)
    let shape = NSBezierPath(roundedRect: body, xRadius: 186 * s, yRadius: 186 * s)
    NSGraphicsContext.saveGraphicsState()
    let shadow = NSShadow()
    shadow.shadowColor = color(0x000000, 0.35)
    shadow.shadowBlurRadius = 24 * s
    shadow.shadowOffset = NSSize(width: 0, height: -10 * s)
    shadow.set()
    color(0x0b0d10).setFill()
    shape.fill()
    NSGraphicsContext.restoreGraphicsState()
    NSGradient(starting: color(0x181c24), ending: color(0x08090b))!.draw(in: shape, angle: -90)
    color(0x2e333b).setStroke()
    shape.lineWidth = max(1, 3 * s)
    shape.stroke()

    // The mark, from the 32-unit favicon (y flipped).
    let unit = 824 * s / 32
    func point(_ x: CGFloat, _ y: CGFloat) -> NSPoint {
        NSPoint(x: body.minX + x * unit, y: body.minY + (32 - y) * unit)
    }
    let mark = NSBezierPath()
    mark.move(to: point(6.5, 23.5))
    for (x, y) in [(11.0, 10.0), (16.0, 19.5), (21.0, 10.0), (25.5, 23.5)] {
        mark.line(to: point(CGFloat(x), CGFloat(y)))
    }
    mark.lineWidth = 2 * unit
    mark.lineCapStyle = .round
    mark.lineJoinStyle = .round
    color(0xeef1f5).setStroke()
    mark.stroke()
    color(0x829cff).setFill()
    for (x, y) in [(6.5, 23.5), (16.0, 19.5), (25.5, 23.5)] {
        let center = point(CGFloat(x), CGFloat(y))
        let r = 2.2 * unit
        NSBezierPath(ovalIn: NSRect(x: center.x - r, y: center.y - r, width: 2 * r, height: 2 * r)).fill()
    }

    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

for points in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(points)x\(points).png" : "icon_\(points)x\(points)@2x.png"
        try draw(size: points * scale).write(to: output.appendingPathComponent(name))
    }
}
