// Native macOS WKWebView proof for the engine used by Tauri's desktop wrapper.
import AppKit
import WebKit

let app = NSApplication.shared
let root = FileManager.default.temporaryDirectory.appendingPathComponent("memorizz-wk-stream-" + UUID().uuidString)
try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let process = Process()
process.executableURL = cwd.appendingPathComponent(".venv/bin/python")
process.arguments = ["tests/integration/streaming_fixture.py", "ui", "8795"]
process.environment = ProcessInfo.processInfo.environment.merging([
    "PYTHONPATH": cwd.appendingPathComponent("src").path,
    "MEMORIZZ_STREAM_FIXTURE_DIR": root.path,
]) { _, new in new }
process.standardOutput = FileHandle.nullDevice
process.standardError = FileHandle.standardError
try process.run()

let configuration = WKWebViewConfiguration()
configuration.websiteDataStore = .nonPersistent()
configuration.userContentController.addUserScript(WKUserScript(source: """
const fixtureFetch = window.fetch;
window.fetch = (input, init = {}) => {
    const headers = new Headers(init.headers || {});
    headers.set('Authorization', 'Bearer stream-fixture-token');
    return fixtureFetch(input, {...init, headers});
};
""", injectionTime: .atDocumentStart, forMainFrameOnly: true))
let webview = WKWebView(frame: NSRect(x: 0, y: 0, width: 1280, height: 900), configuration: configuration)
let window = NSWindow(contentRect: webview.frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
window.title = "Memorizz isolated streaming verification"
window.contentView = webview
window.orderBack(nil)
var phase = 0
var busy = false
var loaded = false
let deadline = Date().addingTimeInterval(50)
func has(_ name: String) -> Bool { FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) }
func touch(_ name: String) { FileManager.default.createFile(atPath: root.appendingPathComponent(name).path, contents: Data()) }
func finish(_ ok: Bool, _ detail: String) {
    print("WKWebView \(ok ? "PASS" : "FAIL"): \(detail); fixture=\(root.path)")
    process.terminate()
    window.close()
    exit(ok ? 0 : 1)
}
Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
    if Date() > deadline {
        finish(false, "deadline exceeded at phase \(phase)")
    }
    if busy { return }
    if !loaded {
        busy = true
        let url = URL(string: "http://127.0.0.1:8795/health")!
        URLSession.shared.dataTask(with: url) { _, response, _ in
            DispatchQueue.main.async {
                busy = false
                if response != nil {
                    loaded = true
                    var request = URLRequest(url: URL(string: "http://127.0.0.1:8795/agents/stream-fixture/playground")!)
                    request.setValue("Bearer stream-fixture-token", forHTTPHeaderField: "Authorization")
                    webview.load(request)
                }
            }
        }.resume()
        return
    }
    busy = true
    let script = """
    JSON.stringify({
      ready: typeof sendMessage === 'function',
      text: document.querySelectorAll('.pg-message-text').length ? Array.from(document.querySelectorAll('.pg-message-text')).at(-1).textContent : '',
      status: document.querySelectorAll('.pg-stream-status').length ? Array.from(document.querySelectorAll('.pg-stream-status')).at(-1).textContent : ''
    })
    """
    webview.evaluateJavaScript(script) { result, error in
        busy = false
        guard let value = result as? String, let data = value.data(using: .utf8),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        let text = state["text"] as? String ?? ""
        let status = state["status"] as? String ?? ""
        if phase == 0 && state["ready"] as? Bool == true {
            phase = 1
            webview.evaluateJavaScript("document.getElementById('pg-query').value='hello'; void sendMessage();")
        } else if phase == 1 && text == "Hello " {
            if has("provider_complete") { finish(false, "first paint came after completion") }
            phase = 2
            touch("release")
        } else if phase == 2 && text.contains("世界") && status == "" {
            phase = 3
            try? FileManager.default.removeItem(at: root.appendingPathComponent("release"))
            try? FileManager.default.removeItem(at: root.appendingPathComponent("provider_closed"))
            webview.evaluateJavaScript("document.getElementById('pg-query').value='stop test'; void sendMessage();")
        } else if phase == 3 && text == "Hello " {
            phase = 4
            webview.evaluateJavaScript("document.getElementById('pg-stop-btn').click();")
        } else if phase == 4 && has("provider_closed") && status.contains("Partial answer retained") {
            if !text.contains("Hello") { finish(false, "partial answer lost") }
            timer.invalidate()
            finish(true, "first paint before completion; two updates; Stop preserves partial and closes provider")
        }
    }
}
app.run()
