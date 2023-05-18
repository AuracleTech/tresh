// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod nn;

use tauri::Window;

// the payload type must implement `Serialize` and `Clone`.
#[derive(Clone, serde::Serialize)]
struct Payload {
    data: String,
}

// init a background process on the command, and emit periodic events only to the window that used the command
#[tauri::command]
async fn init_neural_network(window: Window) {
    tokio::spawn(async move {
        nn::run(&window);
    });
}

#[tokio::main]
async fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![init_neural_network])
        .run(tauri::generate_context!())
        .expect("failed to run app");
}

pub(crate) fn emit(window: &Window, event_name: &str, data: String) {
    window
        .emit(event_name, Payload { data })
        .expect("failed to emit");
}
