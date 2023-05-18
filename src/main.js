const { invoke } = window.__TAURI__.tauri;
const { appWindow, WebviewWindow } = window.__TAURI__.window;

let display;

let start_nn1;
let start_nn2;

window.addEventListener("DOMContentLoaded", () => {
	display = document.getElementById("display");
	start_nn1 = document.getElementById("start-nn1");
	start_nn2 = document.getElementById("start-nn2");

	appWindow.listen("epoch", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	appWindow.listen("result", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	start_nn1.addEventListener("click", () => invoke("start_nn1") && cls());
	start_nn2.addEventListener("click", () => invoke("start_nn2") && cls());
});

function cls() {
	display.innerHTML = "";
}
