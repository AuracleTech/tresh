const { invoke } = window.__TAURI__.tauri;
const { appWindow, WebviewWindow } = window.__TAURI__.window;

let display;

window.addEventListener("DOMContentLoaded", () => {
	display = document.getElementById("display");

	appWindow.listen("epoch", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	appWindow.listen("result", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	invoke("init_neural_network");
});
