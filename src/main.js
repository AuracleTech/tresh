const { invoke } = window.__TAURI__.tauri;
const { appWindow, WebviewWindow } = window.__TAURI__.window;

let display;

let start_nn1;
let start_nn2;
let start_nn3;
let start_nn4;
let start_nn5;
// let fake_login;

window.addEventListener("DOMContentLoaded", async () => {
	display = document.getElementById("display");
	start_nn1 = document.getElementById("start-nn1");
	start_nn2 = document.getElementById("start-nn2");
	start_nn3 = document.getElementById("start-nn3");
	start_nn4 = document.getElementById("start-nn4");
	start_nn5 = document.getElementById("start-nn5");
	// fake_login = document.getElementById("fake-login");

	appWindow.listen("print", (event) => {
		let data = event.payload.data;
		switch (data) {
			case "":
				data = "❌ EMPTY DATA RECEIVED ❌";
				break;
			case "<hr>":
				// do nothing
				break;
			default:
				// replace new lines with <br>
				data = data.replace(/\n/g, "<br>") + "<br>";
				break;
		}
		display.innerHTML += data;
	});

	start_nn4.addEventListener(
		"click",
		() => invoke("start_percepteur") && cls()
	);
	start_nn1.addEventListener("click", () => invoke("start_nn1") && cls());
	start_nn2.addEventListener("click", () => invoke("start_nn2") && cls());
	start_nn3.addEventListener("click", () => invoke("start_nn3") && cls());
	start_nn4.addEventListener("click", () => invoke("start_nn4") && cls());
	start_nn5.addEventListener("click", () => invoke("start_nn5") && cls());
	// fake_login.addEventListener("click", async () => {
	// 	try {
	// 		const state = await invoke("fake_login");
	// 		console.log(state); // { logged_in: true }
	// 	} catch (e) {
	// 		console.error(e);
	// 	}
	// });
});

const cls = () => (display.innerHTML = "");
