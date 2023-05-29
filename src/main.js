const { invoke } = window.__TAURI__.tauri;
const { appWindow, WebviewWindow } = window.__TAURI__.window;

let display;

let start_percepteur;
let start_nn1;
let start_nn2;
let start_nn3;
let test;

window.addEventListener("DOMContentLoaded", async () => {
	display = document.getElementById("display");
	start_percepteur = document.getElementById("start-percepteur");
	start_nn1 = document.getElementById("start-nn1");
	start_nn2 = document.getElementById("start-nn2");
	start_nn3 = document.getElementById("start-nn3");
	test = document.getElementById("test");

	appWindow.listen("print", (event) => {
		display.innerHTML += `${event.payload.data}<br>`;
	});

	start_percepteur.addEventListener(
		"click",
		() => invoke("start_percepteur") && cls()
	);
	start_nn1.addEventListener("click", () => invoke("start_nn1") && cls());
	start_nn2.addEventListener("click", () => invoke("start_nn2") && cls());
	start_nn3.addEventListener("click", () => invoke("start_nn3") && cls());
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

const cls = () => (display.innerHTML = "");
