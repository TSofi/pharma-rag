// Where the backend API lives.
// - Served by FastAPI itself (localhost or Render): same origin -> "".
// - Served by Vercel (or anywhere else): call the Render backend.
(() => {
  const RENDER_API = "https://pharma-rag-smgm.onrender.com";
  const h = location.hostname;
  const sameOrigin = h === "localhost" || h === "127.0.0.1" || h.endsWith(".onrender.com");
  window.API_BASE = sameOrigin ? "" : RENDER_API;
})();
