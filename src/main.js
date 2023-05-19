const { invoke } = window.__TAURI__.tauri;
const { appWindow, WebviewWindow } = window.__TAURI__.window;

let display;

let start_nn1;
let start_nn2;
let test;

window.addEventListener("DOMContentLoaded", async () => {
	display = document.getElementById("display");
	start_nn1 = document.getElementById("start-nn1");
	start_nn2 = document.getElementById("start-nn2");
	test = document.getElementById("test");

	appWindow.listen("epoch", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	appWindow.listen("result", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	start_nn1.addEventListener("click", () => invoke("start_nn1") && cls());
	start_nn2.addEventListener("click", () => invoke("start_nn2") && cls());
	test.addEventListener("click", async () => {
		// appWindow.emit("test", { data: "test", id: 1 });
		try {
			const state = await invoke("login");
			console.log(state);
			// { logged_in: true }
		} catch (e) {
			console.error(e);
		}
	});
});

function cls() {
	display.innerHTML = "";
}
