import "@fontsource-variable/inter";
import "@fontsource-variable/noto-sans-telugu";
import "./app.css";
import { mount } from "svelte";
import App from "./App.svelte";

export default mount(App, { target: document.getElementById("app")! });
