"use strict";

const applicationPrefix = window.location.pathname.replace(/\/docs\/?$/, "");

window.ui = SwaggerUIBundle({
  url: `${applicationPrefix}/openapi.json`,
  dom_id: "#swagger-ui",
  deepLinking: true,
  displayRequestDuration: true,
  presets: [SwaggerUIBundle.presets.apis],
});
