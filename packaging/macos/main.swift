// MemoRizz for Mac: the local MemoRizz UI in its own window.
//
// On launch it shows the UI already running on this Mac, or starts
// `memorizz ui` and shows it once it answers; quitting stops the server it
// started (one it found running is left alone). When MemoRizz isn't
// installed, it offers to install it with uv. External links open in the
// default browser, and downloads land in ~/Downloads. The server's output
// goes to ~/Library/Logs/MemoRizz/ui.log. MEMORIZZ_UI_PORT picks another
// port (default 8765) and MEMORIZZ_APP_COMMAND a `memorizz` to run.
//
// Build with packaging/macos/build.sh.

import AppKit
import WebKit

let uiPort = Int(ProcessInfo.processInfo.environment["MEMORIZZ_UI_PORT"] ?? "") ?? 8765
let uiURL = URL(string: "http://127.0.0.1:\(uiPort)/")!
let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory())
let logURL = home.appendingPathComponent("Library/Logs/MemoRizz/ui.log")
let installScript = """
set -e
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (https://docs.astral.sh/uv/)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "Installing MemoRizz..."
uv tool install --python 3.12 'memorizz[ui,mcp]'
"""

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate,
    WKDownloadDelegate
{
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusView: NSStackView!
    private var statusLabel: NSTextField!
    private var detailLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var actionButton: NSButton!
    private var action: (() -> Void)?
    private var server: Process?
    private var stopping = false
    private var environment: [String: String] = ProcessInfo.processInfo.environment

    // MARK: Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        buildWindow()
        start()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) { stopServer() }

    // MARK: Starting the UI

    private func start() {
        showStatus("Starting MemoRizz…", detail: nil, busy: true)
        DispatchQueue.global(qos: .userInitiated).async {
            self.environment = Self.loginEnvironment()
            if Self.isUp() {
                DispatchQueue.main.async { self.showUI() }
                return
            }
            guard let memorizz = self.findMemorizz() else {
                DispatchQueue.main.async { self.offerInstall() }
                return
            }
            self.launch(memorizz)
        }
    }

    /// The user's login PATH, with the places uv, Homebrew and pip put tools.
    private static func loginEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let shell = Process()
        shell.executableURL = URL(fileURLWithPath: "/bin/zsh")
        shell.arguments = ["-lc", "printf %s \"$PATH\""]
        let pipe = Pipe()
        shell.standardOutput = pipe
        shell.standardError = FileHandle.nullDevice
        var paths: [String] = []
        if (try? shell.run()) != nil {
            let deadline = Date().addingTimeInterval(5)
            while shell.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
            if shell.isRunning { shell.terminate() }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            paths = (String(data: data, encoding: .utf8) ?? "").split(separator: ":").map(String.init)
        }
        for extra in ["\(home.path)/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        where !paths.contains(extra) {
            paths.append(extra)
        }
        env["PATH"] = paths.joined(separator: ":")
        return env
    }

    private func findMemorizz() -> URL? {
        if let chosen = ProcessInfo.processInfo.environment["MEMORIZZ_APP_COMMAND"], !chosen.isEmpty {
            return FileManager.default.isExecutableFile(atPath: chosen) ? URL(fileURLWithPath: chosen) : nil
        }
        let paths = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for directory in paths {
            let candidate = URL(fileURLWithPath: directory).appendingPathComponent("memorizz")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// Anything answering HTTP on the port: the UI, started here or elsewhere.
    private static func isUp() -> Bool {
        var request = URLRequest(url: uiURL)
        request.timeoutInterval = 1
        let done = DispatchSemaphore(value: 0)
        var answered = false
        URLSession.shared.dataTask(with: request) { _, response, _ in
            answered = response is HTTPURLResponse
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 2)
        return answered
    }

    private func launch(_ memorizz: URL) {
        let process = Process()
        process.executableURL = memorizz
        process.arguments = ["ui", "--port", String(uiPort)]
        var env = environment
        env["MEMORIZZ_UI_OPEN_BROWSER"] = "false"  // this window shows it
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        process.currentDirectoryURL = home
        let log = Self.openLog()
        process.standardOutput = log
        process.standardError = log
        process.terminationHandler = { finished in
            DispatchQueue.main.async {
                if !self.stopping && self.server === finished {
                    self.showProblem(
                        "MemoRizz stopped (exit status \(finished.terminationStatus)).",
                        retry: "Start again")
                }
            }
        }
        do {
            try process.run()
        } catch {
            DispatchQueue.main.async {
                self.showProblem("Could not run \(memorizz.path): \(error.localizedDescription)",
                    retry: "Try again")
            }
            return
        }
        server = process
        let deadline = Date().addingTimeInterval(180)
        while Date() < deadline {
            if Self.isUp() {
                DispatchQueue.main.async { self.showUI() }
                return
            }
            if !process.isRunning { return }  // the termination handler says why
            Thread.sleep(forTimeInterval: 0.3)
        }
        DispatchQueue.main.async {
            self.showProblem("MemoRizz did not start within three minutes.", retry: "Try again")
        }
    }

    /// A line from the app itself in the server log.
    private static func note(_ line: String) {
        guard let handle = try? FileHandle(forWritingTo: logURL) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("MemoRizz app: \(line)\n".utf8))
    }

    private static func openLog() -> FileHandle {
        try? FileManager.default.createDirectory(
            at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        return (try? FileHandle(forWritingTo: logURL)) ?? FileHandle.nullDevice
    }

    private func stopServer() {
        stopping = true
        guard let process = server, process.isRunning else { return }
        process.interrupt()  // Ctrl+C: the UI stops at once
        let deadline = Date().addingTimeInterval(6)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        if process.isRunning { process.terminate() }
    }

    // MARK: Installing

    private func offerInstall() {
        showStatus(
            "MemoRizz isn't installed on this Mac yet.",
            detail:
                "Install it with uv, which also sets up the Python it needs. This takes a minute or two.",
            busy: false)
        showAction("Install MemoRizz") { self.install() }
    }

    private func install() {
        showStatus(
            "Installing MemoRizz…", detail: "uv tool install --python 3.12 'memorizz[ui,mcp]'",
            busy: true)
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = ["-lc", installScript]
            process.environment = self.environment
            process.currentDirectoryURL = home
            let log = Self.openLog()
            process.standardOutput = log
            process.standardError = log
            do {
                try process.run()
                process.waitUntilExit()
            } catch {}
            DispatchQueue.main.async {
                if process.terminationReason == .exit && process.terminationStatus == 0 {
                    self.start()
                } else {
                    self.showProblem("Installing MemoRizz failed.", retry: "Try again")
                }
            }
        }
    }

    // MARK: Window

    private func buildWindow() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 840),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "MemoRizz"
        window.minSize = NSSize(width: 860, height: 560)
        window.center()
        window.setFrameAutosaveName("MemoRizzMain")
        window.tabbingMode = .disallowed

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.applicationNameForUserAgent = "MemoRizz-Mac"
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.isHidden = true

        let icon = NSImageView(image: NSApp.applicationIconImage)
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.widthAnchor.constraint(equalToConstant: 96).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 96).isActive = true
        statusLabel = NSTextField(labelWithString: "")
        statusLabel.font = .systemFont(ofSize: 17, weight: .semibold)
        statusLabel.alignment = .center
        detailLabel = NSTextField(wrappingLabelWithString: "")
        detailLabel.font = .systemFont(ofSize: 13)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.alignment = .center
        detailLabel.preferredMaxLayoutWidth = 520
        detailLabel.isSelectable = true
        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        actionButton = NSButton(title: "", target: self, action: #selector(runAction))
        actionButton.bezelStyle = .rounded
        actionButton.keyEquivalent = "\r"
        statusView = NSStackView(views: [icon, statusLabel, detailLabel, spinner, actionButton])
        statusView.orientation = .vertical
        statusView.alignment = .centerX
        statusView.spacing = 12

        let content = NSView()
        for view in [webView!, statusView!] as [NSView] {
            view.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(view)
        }
        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            webView.topAnchor.constraint(equalTo: content.topAnchor),
            webView.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            statusView.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            statusView.centerYAnchor.constraint(equalTo: content.centerYAnchor),
            statusView.widthAnchor.constraint(lessThanOrEqualToConstant: 560),
        ])
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
    }

    private func showStatus(_ title: String, detail: String?, busy: Bool) {
        webView.isHidden = true
        statusView.isHidden = false
        statusLabel.stringValue = title
        detailLabel.stringValue = detail ?? ""
        detailLabel.isHidden = detail == nil
        spinner.isHidden = !busy
        if busy {
            spinner.startAnimation(nil)
        } else {
            spinner.stopAnimation(nil)
        }
        actionButton.isHidden = true
        action = nil
    }

    private func showAction(_ title: String, _ run: @escaping () -> Void) {
        actionButton.title = title
        actionButton.isHidden = false
        action = run
    }

    private func showProblem(_ title: String, retry: String) {
        showStatus(title, detail: Self.logTail(), busy: false)
        showAction(retry) { self.start() }
    }

    private static func logTail() -> String {
        let text = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true).suffix(6)
        let tail = lines.joined(separator: "\n")
        return tail.isEmpty ? "The log is at \(logURL.path)." : tail
    }

    @objc private func runAction() { action?() }

    private func showUI() {
        statusView.isHidden = true
        spinner.stopAnimation(nil)
        webView.isHidden = false
        webView.load(URLRequest(url: uiURL))
    }

    // MARK: Menus

    private func buildMenu() {
        let main = NSMenu()
        func submenu(_ title: String, _ items: [NSMenuItem]) {
            let holder = NSMenuItem()
            let menu = NSMenu(title: title)
            items.forEach(menu.addItem)
            holder.submenu = menu
            main.addItem(holder)
        }
        func item(_ title: String, _ action: Selector?, _ key: String, _ mods: NSEvent.ModifierFlags = .command)
            -> NSMenuItem
        {
            let entry = NSMenuItem(title: title, action: action, keyEquivalent: key)
            entry.keyEquivalentModifierMask = mods
            return entry
        }
        submenu(
            "MemoRizz",
            [
                item(
                    "About MemoRizz",
                    #selector(NSApplication.orderFrontStandardAboutPanel(_:)), ""),
                .separator(),
                item("Hide MemoRizz", #selector(NSApplication.hide(_:)), "h"),
                item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", [.command, .option]),
                item("Show All", #selector(NSApplication.unhideAllApplications(_:)), ""),
                .separator(),
                item("Quit MemoRizz", #selector(NSApplication.terminate(_:)), "q"),
            ])
        submenu(
            "Edit",
            [
                item("Undo", Selector(("undo:")), "z"),
                item("Redo", Selector(("redo:")), "z", [.command, .shift]),
                .separator(),
                item("Cut", #selector(NSText.cut(_:)), "x"),
                item("Copy", #selector(NSText.copy(_:)), "c"),
                item("Paste", #selector(NSText.paste(_:)), "v"),
                item("Select All", #selector(NSText.selectAll(_:)), "a"),
            ])
        let reload = item("Reload", #selector(reloadPage), "r")
        let back = item("Back", #selector(goBack), "[")
        let forward = item("Forward", #selector(goForward), "]")
        let browser = item("Open in Browser", #selector(openInBrowser), "o", [.command, .shift])
        let log = item("Show Server Log", #selector(showLog), "")
        for entry in [reload, back, forward, browser, log] { entry.target = self }
        submenu("View", [reload, back, forward, .separator(), browser, log])
        submenu(
            "Window",
            [
                item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"),
                item("Zoom", #selector(NSWindow.performZoom(_:)), ""),
            ])
        NSApp.mainMenu = main
    }

    @objc private func reloadPage() {
        if webView.isHidden {
            start()
        } else {
            webView.reload()
        }
    }
    @objc private func goBack() { webView.goBack() }
    @objc private func goForward() { webView.goForward() }
    @objc private func openInBrowser() {
        NSWorkspace.shared.open(webView.url ?? uiURL)
    }
    @objc private func showLog() { NSWorkspace.shared.open(logURL) }

    // MARK: Links, downloads and dialogs

    private static func isLocal(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else { return true }
        if scheme == "about" || scheme == "blob" || scheme == "data" { return true }
        return (scheme == "http" || scheme == "https")
            && ["127.0.0.1", "localhost", "::1"].contains(url.host?.lowercased() ?? "")
    }

    func webView(
        _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        if let url = navigationAction.request.url, !Self.isLocal(url) {
            NSWorkspace.shared.open(url)  // other sites open in the browser
            decisionHandler(.cancel)
            return
        }
        decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
    }

    func webView(
        _ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        let disposition =
            (navigationResponse.response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Disposition")?.lowercased() ?? ""
        if !navigationResponse.canShowMIMEType || disposition.hasPrefix("attachment") {
            decisionHandler(.download)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(
        _ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload
    ) {
        download.delegate = self
    }

    func download(
        _ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        let folder = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        let name = suggestedFilename.isEmpty ? "download" : suggestedFilename
        var target = folder.appendingPathComponent(name)
        var copy = 1
        while FileManager.default.fileExists(atPath: target.path) {
            let base = (name as NSString).deletingPathExtension
            let ext = (name as NSString).pathExtension
            target = folder.appendingPathComponent(ext.isEmpty ? "\(base) \(copy)" : "\(base) \(copy).\(ext)")
            copy += 1
        }
        completionHandler(target)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            if FileManager.default.fileExists(atPath: target.path) {
                NSWorkspace.shared.activateFileViewerSelecting([target])
            }
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Self.note("showing \(webView.url?.absoluteString ?? "?") (\(webView.title ?? ""))")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        Self.note("the page's web process stopped; reloading")
        webView.reload()
    }

    func webView(
        _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error
    ) {
        let code = (error as NSError).code
        if code == NSURLErrorCancelled || code == 102 { return }  // a download, or cancelled
        if !Self.isUp() { showProblem("MemoRizz isn't answering.", retry: "Start again") }
    }

    func webView(
        _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if let url = navigationAction.request.url {
            if Self.isLocal(url) {
                webView.load(navigationAction.request)
            } else {
                NSWorkspace.shared.open(url)
            }
        }
        return nil
    }

    func webView(
        _ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = message
        alert.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(
        _ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    func webView(
        _ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
        defaultText: String?, initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (String?) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) {
            completionHandler($0 == .alertFirstButtonReturn ? field.stringValue : nil)
        }
    }

    func webView(
        _ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.beginSheetModal(for: window) { completionHandler($0 == .OK ? panel.urls : nil) }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
