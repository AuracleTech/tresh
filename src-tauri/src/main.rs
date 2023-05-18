// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod nn1;
mod nn2;

use tauri::Window;

// the payload type must implement `Serialize` and `Clone`.
#[derive(Clone, serde::Serialize)]
struct Payload {
    data: String,
}

#[tauri::command]
async fn start_nn1(window: Window) {
    nn1::run(&window)
}

#[tauri::command]
async fn start_nn2(window: Window) {
    nn2::run(&window)
}

#[tokio::main]
async fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![start_nn1, start_nn2])
        .run(tauri::generate_context!())
        .expect("failed to run app");
}

const EPOCHS_PER_UPDATE: usize = 250;
pub(crate) fn emit(window: &Window, event_name: &str, data: String) {
    window
        .emit(event_name, Payload { data })
        .expect("failed to emit");
}
