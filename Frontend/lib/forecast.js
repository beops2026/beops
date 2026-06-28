/**
 * Forecast utilities for LoadSense Grid Substation Forecasting System
 */

/**
 * Convert a JS Date object to a Unix epoch timestamp (integer seconds).
 * The backend requires dates as Unix epoch integers, never date strings.
 * @param {Date} date
 * @returns {number} Unix epoch in seconds
 */
export function toEpoch(date) {
  return Math.floor(date.getTime() / 1000);
}

/**
 * Convert a JS Date to midnight (00:00:00) epoch for that day.
 * Used for start_date when sending to the backend.
 * @param {Date} date
 * @returns {number} Unix epoch in seconds at midnight
 */
export function toMidnightEpoch(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return Math.floor(d.getTime() / 1000);
}

/**
 * Convert a JS Date to end-of-day (23:59:59) epoch for that day.
 * @param {Date} date
 * @returns {number} Unix epoch in seconds at end of day
 */
export function toEndOfDayEpoch(date) {
  const d = new Date(date);
  d.setHours(23, 59, 59, 999);
  return Math.floor(d.getTime() / 1000);
}

/**
 * Format an epoch timestamp for display in chart labels.
 * @param {number} epoch - Unix epoch in seconds
 * @param {"hourly"|"weekly"|"monthly"|"quarterly"} forecastType
 * @returns {string} Formatted label
 */
export function formatEpochLabel(epoch, forecastType) {
  const date = new Date(epoch * 1000);

  switch (forecastType) {
    case "hourly": {
      // e.g. "14:00"
      const h = date.getHours().toString().padStart(2, "0");
      return `${h}:00`;
    }
    case "weekly": {
      // Hourly resolution across the range — e.g. "Mon 14:00"
      const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
      const h = date.getHours().toString().padStart(2, "0");
      return `${days[date.getDay()]} ${h}:00`;
    }
    case "monthly": {
      // e.g. "23 Jun"
      const months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
      ];
      return `${date.getDate()} ${months[date.getMonth()]}`;
    }
    case "quarterly": {
      // e.g. "Jun 2026"
      const months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
      ];
      return `${months[date.getMonth()]} ${date.getFullYear()}`;
    }
    default:
      return date.toLocaleDateString();
  }
}

/**
 * Format an epoch timestamp as a full readable datetime string for tooltips.
 * @param {number} epoch
 * @param {"hourly"|"weekly"|"monthly"|"quarterly"} forecastType
 * @returns {string}
 */
export function formatEpochFull(epoch, forecastType) {
  const date = new Date(epoch * 1000);
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const day = date.getDate();
  const month = months[date.getMonth()];
  const year = date.getFullYear();

  if (forecastType === "hourly" || forecastType === "weekly") {
    const h = date.getHours().toString().padStart(2, "0");
    return `${day} ${month} ${year}, ${h}:00`;
  }
  if (forecastType === "quarterly") {
    return `${month} ${year}`;
  }
  return `${day} ${month} ${year}`;
}

/**
 * Validate date range based on forecast type.
 * Returns null if valid, or an error string if invalid.
 * @param {Date} startDate
 * @param {Date} endDate
 * @param {"hourly"|"weekly"|"monthly"|"quarterly"} forecastType
 * @returns {string|null}
 */
export function validateDateRange(startDate, endDate, forecastType) {
  if (!startDate || !endDate) {
    return "Please select both start and end dates.";
  }

  const start = new Date(startDate);
  const end = new Date(endDate);
  start.setHours(0, 0, 0, 0);
  end.setHours(0, 0, 0, 0);

  if (end < start) {
    return "End date must be after start date.";
  }

  const diffDays = Math.round((end - start) / (1000 * 60 * 60 * 24));

  switch (forecastType) {
    case "hourly":
      if (diffDays !== 0) {
        return "Hourly forecast: start and end date must be the same day.";
      }
      break;
    case "weekly":
      if (diffDays < 1) {
        return "Weekly forecast: please select at least 2 days.";
      }
      if (diffDays > 14) {
        return "Weekly forecast: maximum range is 14 days.";
      }
      break;
    case "monthly":
      if (diffDays < 7) {
        return "Monthly forecast: please select at least 7 days.";
      }
      if (diffDays > 92) {
        return "Monthly forecast: maximum range is 92 days (about 3 months).";
      }
      break;
    case "quarterly":
      if (diffDays > 108) {
        return "Quarterly forecast: maximum supported range is 108 days (Open-Meteo weather API limit).";
      }
      break;
    default:
      return "Unknown forecast type.";
  }

  return null;
}

/**
 * Compute summary statistics from a predictions array.
 * @param {{ time: number, value: number }[]} predictions
 * @returns {{ average: number, maximum: number, minimum: number, total: number, trend: string }}
 */
export function computeSummary(predictions) {
  if (!predictions || predictions.length === 0) {
    return { average: 0, maximum: 0, minimum: 0, total: 0, trend: "stable" };
  }

  const values = predictions.map((p) => p.value);
  const total = values.reduce((a, b) => a + b, 0);
  const average = total / values.length;
  const maximum = Math.max(...values);
  const minimum = Math.min(...values);

  let trend = "stable";
  if (values.length >= 2) {
    const delta = values[values.length - 1] - values[0];
    if (delta > 0) trend = "rising";
    else if (delta < 0) trend = "falling";
  }

  return {
    average: Math.round(average * 100) / 100,
    maximum: Math.round(maximum * 100) / 100,
    minimum: Math.round(minimum * 100) / 100,
    total: Math.round(total * 100) / 100,
    trend,
  };
}

/**
 * Parse AI report text that may be either:
 * 1. A structured JSON object with keys: executive_summary, key_observations, operational_risks, plan_of_action, next_steps
 * 2. A plain text string with section headers
 * 
 * Returns a normalized object with those five keys.
 * @param {string|object} report
 * @returns {{ executive_summary: string, key_observations: string[], operational_risks: string[], plan_of_action: string[], next_steps: string[] }}
 */
export function parseReport(report) {
  // Already a structured object from LLM
  if (typeof report === "object" && report !== null) {
    return {
      executive_summary: report.executive_summary || "",
      key_observations: ensureArray(report.key_observations),
      operational_risks: ensureArray(report.operational_risks),
      plan_of_action: ensureArray(report.plan_of_action),
      next_steps: ensureArray(report.next_steps),
    };
  }

  // Plain text — parse section headers
  if (typeof report === "string") {
    return parseTextReport(report);
  }

  return {
    executive_summary: "Report data unavailable.",
    key_observations: [],
    operational_risks: [],
    plan_of_action: [],
    next_steps: [],
  };
}

function ensureArray(val) {
  if (Array.isArray(val)) return val;
  if (typeof val === "string") return [val];
  return [];
}

function parseTextReport(text) {
  const sections = {
    executive_summary: "",
    key_observations: [],
    operational_risks: [],
    plan_of_action: [],
    next_steps: [],
  };

  const headerMap = {
    "Executive Summary": "executive_summary",
    "Key Observations": "key_observations",
    "Operational Risks": "operational_risks",
    "Plan of Action": "plan_of_action",
    "Next Steps": "next_steps",
  };

  const lines = text.split("\n");
  let currentSection = null;
  let summaryLines = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Check if line is a section header
    const matchedHeader = Object.keys(headerMap).find((h) =>
      trimmed.startsWith(h)
    );

    if (matchedHeader) {
      currentSection = headerMap[matchedHeader];
      continue;
    }

    if (!currentSection) {
      // Before any header — treat as executive summary
      summaryLines.push(trimmed);
      continue;
    }

    if (currentSection === "executive_summary") {
      summaryLines.push(trimmed);
    } else {
      // Strip leading "- " or "• " bullet markers
      const content = trimmed.replace(/^[-•]\s*/, "");
      if (content) sections[currentSection].push(content);
    }
  }

  sections.executive_summary = summaryLines.join(" ");
  return sections;
}

/**
 * Get a human-readable label for a forecast type.
 * @param {"hourly"|"weekly"|"monthly"|"quarterly"} type
 * @returns {string}
 */
export function getForecastTypeLabel(type) {
  const labels = {
    hourly: "Hourly Forecast",
    weekly: "Weekly Forecast",
    monthly: "Monthly Forecast",
    quarterly: "Quarterly Forecast",
  };
  return labels[type] || type;
}

/**
 * Get the unit label for chart Y-axis depending on forecast type.
 * Hourly = MW (individual hour values), others = MWh/day or MWh/month totals.
 */
export function getValueUnit(forecastType) {
  // hourly and weekly are instantaneous hourly load (MW); monthly/quarterly
  // are aggregated energy totals (MWh).
  if (forecastType === "hourly" || forecastType === "weekly") return "MW";
  return "MWh";
}
