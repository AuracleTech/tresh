// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod nn1;
mod nn2;

use serde::Serialize;
use std::sync::Mutex;
use tauri::{State, Window};

#[derive(Serialize, Clone)]
pub(crate) struct AuthState {
    // #[serde(skip_serializing)]
    // token: Option<String>,
    logged_in: bool,
}

#[tauri::command]
fn login(state_mutex: State<Mutex<AuthState>>) -> Result<AuthState, String> {
    println!("Logging in");
    let mut state = state_mutex.lock().expect("failed to lock state");
    state.logged_in = true;
    Ok(state.clone())
}

// the payload type must implement `Serialize` and `Clone`.
#[derive(Clone, Serialize)]
struct Payload {
    data: String,
}

#[tauri::command]
async fn start_nn1(window: Window) {
    nn1::run(&window);
    let id = window.listen("test", |event| {
        println!("got window event-name with payload {:?}", event.payload());
    });
    window.unlisten(id);
}

#[tauri::command]
async fn start_nn2(window: Window) {
    nn2::run(&window)
}

#[tokio::main]
async fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(AuthState {
            // token: None,
            logged_in: false,
        }))
        .invoke_handler(tauri::generate_handler![start_nn1, start_nn2, login])
        .run(tauri::generate_context!())
        .expect("failed to run app");
}

const EPOCHS_PER_UPDATE: usize = 250;
pub(crate) fn emit<T: ToString>(window: &Window, event_name: &str, data: T) {
    let payload = Payload {
        data: data.to_string(),
    };
    if let Err(err) = window.emit(event_name, payload) {
        eprintln!("Failed to emit event: {}", err);
    }
}
