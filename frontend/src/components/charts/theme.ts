// Chart chrome for the dark surface (validated palette, see globals.css).
export const C = {
  surface: "#1a1a19", grid: "#2c2c2a", axis: "#383835", muted: "#898781", ink: "#f0efec", ink2: "#c3c2b7",
  up: "#199e70", down: "#e66767", accent: "#e3b341",
  good: "#0ca30c", bad: "#d03b3b",
  series: ["#3987e5", "#d95926", "#199e70", "#c98500"], // fixed order, never cycled
  diverging: { neg: "#e66767", mid: "#383835", pos: "#3987e5" },
};

export const nyFormatter = (t: number) =>
  new Date(t * 1000).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false }) + " NY";
