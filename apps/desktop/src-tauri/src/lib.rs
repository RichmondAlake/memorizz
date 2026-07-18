use tauri::menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // The default Tauri shell ships without an app menu on macOS, which
        // means the webview never receives ⌘R, ⌘Q, copy/paste accelerators,
        // etc. We register a minimal native menu here so the desktop app
        // behaves like every other Mac app — most importantly, ⌘R reloads
        // the embedded UI so iterating on the FastAPI server doesn't require
        // quitting and re-launching the bundle.
        .menu(|app| {
            let app_submenu = SubmenuBuilder::new(app, "Memorizz")
                .item(&PredefinedMenuItem::about(app, None, None)?)
                .separator()
                .item(&PredefinedMenuItem::hide(app, None)?)
                .item(&PredefinedMenuItem::hide_others(app, None)?)
                .item(&PredefinedMenuItem::show_all(app, None)?)
                .separator()
                .item(&PredefinedMenuItem::quit(app, None)?)
                .build()?;

            let edit_submenu = SubmenuBuilder::new(app, "Edit")
                .item(&PredefinedMenuItem::undo(app, None)?)
                .item(&PredefinedMenuItem::redo(app, None)?)
                .separator()
                .item(&PredefinedMenuItem::cut(app, None)?)
                .item(&PredefinedMenuItem::copy(app, None)?)
                .item(&PredefinedMenuItem::paste(app, None)?)
                .item(&PredefinedMenuItem::select_all(app, None)?)
                .build()?;

            let reload = MenuItemBuilder::with_id("reload", "Reload")
                .accelerator("CmdOrCtrl+R")
                .build(app)?;
            let force_reload = MenuItemBuilder::with_id("force_reload", "Force Reload")
                .accelerator("CmdOrCtrl+Shift+R")
                .build(app)?;
            let toggle_devtools =
                MenuItemBuilder::with_id("toggle_devtools", "Toggle Developer Tools")
                    .accelerator("CmdOrCtrl+Alt+I")
                    .build(app)?;

            let view_submenu = SubmenuBuilder::new(app, "View")
                .item(&reload)
                .item(&force_reload)
                .separator()
                .item(&toggle_devtools)
                .build()?;

            let window_submenu = SubmenuBuilder::new(app, "Window")
                .item(&PredefinedMenuItem::minimize(app, None)?)
                .item(&PredefinedMenuItem::close_window(app, None)?)
                .build()?;

            MenuBuilder::new(app)
                .item(&app_submenu)
                .item(&edit_submenu)
                .item(&view_submenu)
                .item(&window_submenu)
                .build()
        })
        .on_menu_event(|app_handle, event| {
            let Some(window) = app_handle.get_webview_window("main") else {
                return;
            };
            match event.id().as_ref() {
                "reload" => {
                    let _ = window.eval("window.location.reload()");
                }
                "force_reload" => {
                    // `reload(true)` is a no-op in modern WebKit but the
                    // explicit cache-buster query string forces a fresh GET
                    // for the embedded URL.
                    let _ = window.eval(
                        "window.location.replace(\
                            window.location.pathname + '?_=' + Date.now() + window.location.hash\
                        )",
                    );
                }
                "toggle_devtools" => {
                    #[cfg(any(debug_assertions, feature = "devtools"))]
                    {
                        if window.is_devtools_open() {
                            window.close_devtools();
                        } else {
                            window.open_devtools();
                        }
                    }
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
