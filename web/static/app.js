const jsonOutput = (target, payload) => {
  const element = document.getElementById(target);
  if (!element) return;
  element.textContent = JSON.stringify(payload, null, 2);
};

const safeParseJson = (value) => {
  if (!value) return { parsed: null, error: null };
  try {
    return { parsed: JSON.parse(value), error: null };
  } catch (error) {
    return { parsed: null, error: "Invalid JSON. Please check formatting." };
  }
};

const fetchJson = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  try {
    return { status: response.status, data: JSON.parse(text) };
  } catch (error) {
    return { status: response.status, data: text };
  }
};

const getValue = (id) => document.getElementById(id)?.value?.trim();

const scrollToTarget = (selector) => {
  const target = document.querySelector(selector);
  if (target) {
    target.scrollIntoView({ behavior: "smooth" });
  }
};

document.querySelectorAll("[data-scroll]").forEach((button) => {
  button.addEventListener("click", () => scrollToTarget(button.dataset.scroll));
});

document.getElementById("health-check").addEventListener("click", async () => {
  const result = await fetchJson("/health");
  jsonOutput("health-output", result);
});

document.getElementById("agri-sample").addEventListener("click", () => {
  document.getElementById("agri-name").value = "Delta Parcel 09";
  document.getElementById("agri-crop").value = "Rice";
  document.getElementById("agri-start").value = "2024-01-01";
  document.getElementById("agri-end").value = "2024-02-01";
  document.getElementById("agri-geometry").value = JSON.stringify(
    {
      type: "Polygon",
      coordinates: [
        [
          [72.876, 19.074],
          [72.888, 19.074],
          [72.888, 19.083],
          [72.876, 19.083],
          [72.876, 19.074],
        ],
      ],
    },
    null,
    2
  );
});

document.getElementById("agri-start-btn").addEventListener("click", async () => {
  const geometryText = getValue("agri-geometry");
  const geometryResult = safeParseJson(geometryText);
  if (geometryResult.error) {
    jsonOutput("agri-output", { error: geometryResult.error });
    return;
  }
  const payload = {
    aoi_name: getValue("agri-name"),
    crop_type: getValue("agri-crop") || "Unknown",
    start_date: getValue("agri-start") || null,
    end_date: getValue("agri-end") || null,
    geometry: geometryResult.parsed,
  };

  const result = await fetchJson("/api/analysis/agri/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (result.data?.task_id) {
    document.getElementById("agri-task").value = result.data.task_id;
  }
  jsonOutput("agri-output", result);
});

document.getElementById("agri-status-btn").addEventListener("click", async () => {
  const taskId = getValue("agri-task");
  if (!taskId) return;
  const result = await fetchJson(`/api/analysis/agri/results/${taskId}`);
  jsonOutput("agri-output", result);
});

document.getElementById("stab-sample").addEventListener("click", () => {
  document.getElementById("stab-lat").value = "28.6139";
  document.getElementById("stab-lon").value = "77.2090";
});

document.getElementById("stab-start-btn").addEventListener("click", async () => {
  const payload = {
    coordinate: {
      latitude: Number(getValue("stab-lat")),
      longitude: Number(getValue("stab-lon")),
    },
  };
  const result = await fetchJson("/api/displacement/analysis/stability/predict/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (result.data?.task_id) {
    document.getElementById("stab-task").value = result.data.task_id;
  }
  jsonOutput("stab-output", result);
});

document.getElementById("stab-status-btn").addEventListener("click", async () => {
  const taskId = getValue("stab-task");
  if (!taskId) return;
  const result = await fetchJson(`/api/displacement/analysis/stability/predict/results/${taskId}`);
  jsonOutput("stab-output", result);
});

document.getElementById("fac-sample").addEventListener("click", () => {
  document.getElementById("fac-country").value = "India";
  document.getElementById("fac-sector").value = "Chemical industry";
  document.getElementById("fac-pollutant").value = "Ammonia";
  document.getElementById("fac-year").value = "2020";
  document.getElementById("fac-limit").value = "25";
});

document.getElementById("fac-search-btn").addEventListener("click", async () => {
  const params = new URLSearchParams();
  const entries = {
    country: getValue("fac-country"),
    sector: getValue("fac-sector"),
    pollutant: getValue("fac-pollutant"),
    year: getValue("fac-year"),
    limit: getValue("fac-limit") || "25",
  };

  Object.entries(entries).forEach(([key, value]) => {
    if (value) params.append(key, value);
  });

  const result = await fetchJson(`/api/facilities/search?${params.toString()}`);
  jsonOutput("fac-output", result);
});

document.getElementById("sim-sample").addEventListener("click", () => {
  document.getElementById("sim-site").value = "ind_site_taloja_44";
  document.getElementById("sim-type").value = "flood";
  document.getElementById("sim-mag").value = "2.5";
  document.getElementById("sim-unit").value = "meters_above_base";
  document.getElementById("sim-meteo").value = JSON.stringify(
    {
      wind_speed_ms: 5.0,
      wind_direction_deg: 180.0,
      temperature_c: 25.0,
    },
    null,
    2
  );
});

document.getElementById("sim-start-btn").addEventListener("click", async () => {
  const meteoText = getValue("sim-meteo");
  const meteoResult = safeParseJson(meteoText);
  if (meteoResult.error) {
    jsonOutput("sim-output", { error: meteoResult.error });
    return;
  }
  const payload = {
    site_id: getValue("sim-site"),
    calamity_type: getValue("sim-type"),
    magnitude: Number(getValue("sim-mag")),
    unit: getValue("sim-unit"),
    meteorological_conditions: meteoResult.parsed,
  };

  const result = await fetchJson("/api/simulate/calamity", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (result.data?.simulation_id) {
    document.getElementById("sim-id").value = result.data.simulation_id;
  }
  jsonOutput("sim-output", result);
});

document.getElementById("sim-risk-btn").addEventListener("click", async () => {
  const simId = getValue("sim-id");
  if (!simId) return;
  const result = await fetchJson(`/api/simulate/risk-profile/${simId}`);
  jsonOutput("sim-output", result);
});

document.getElementById("met-sample").addEventListener("click", () => {
  document.getElementById("met-lat").value = "19.0760";
  document.getElementById("met-lon").value = "72.8777";
  document.getElementById("met-hours").value = "24";
});

document.getElementById("met-current-btn").addEventListener("click", async () => {
  const lat = getValue("met-lat");
  const lon = getValue("met-lon");
  const result = await fetchJson(`/api/meteorological/current?lat=${lat}&lon=${lon}`);
  jsonOutput("met-output", result);
});

document.getElementById("met-forecast-btn").addEventListener("click", async () => {
  const lat = getValue("met-lat");
  const lon = getValue("met-lon");
  const hours = getValue("met-hours") || "24";
  const result = await fetchJson(`/api/meteorological/forecast?lat=${lat}&lon=${lon}&hours=${hours}`);
  jsonOutput("met-output", result);
});

document.getElementById("ter-sample").addEventListener("click", () => {
  document.getElementById("ter-lat").value = "35.6895";
  document.getElementById("ter-lon").value = "139.6917";
});

const terrainFetch = async (endpoint) => {
  const lat = getValue("ter-lat");
  const lon = getValue("ter-lon");
  const result = await fetchJson(`/api/terrain/${endpoint}?lat=${lat}&lon=${lon}`);
  jsonOutput("ter-output", result);
};

document.getElementById("ter-elev-btn").addEventListener("click", () => terrainFetch("elevation"));
document.getElementById("ter-slope-btn").addEventListener("click", () => terrainFetch("slope"));
document.getElementById("ter-rough-btn").addEventListener("click", () => terrainFetch("roughness"));
document.getElementById("ter-flow-btn").addEventListener("click", () => terrainFetch("flow-direction"));
