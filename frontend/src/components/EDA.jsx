import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

const CHURN_BY_CONTRACT = {
  labels: ["Month-to-month", "One year", "Two year"],
  data: [43, 11, 3],
  colors: ["#E24B4A", "#378ADD", "#639922"],
};

const CHURN_BY_INTERNET = {
  labels: ["Fiber optic", "DSL", "No service"],
  data: [42, 19, 7],
  colors: ["#E24B4A", "#378ADD", "#639922"],
};

const DRIVERS = [
  { label: "Month-to-month contract", pct: 88 },
  { label: "Fiber optic internet", pct: 68 },
  { label: "No online security", pct: 55 },
  { label: "Monthly charges > $65", pct: 48 },
  { label: "No tech support", pct: 41 },
  { label: "Tenure < 12 months", pct: 38 },
];

function BarChart({ id, labels, data, colors }) {
  const ref = useRef(null);
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ data, backgroundColor: colors, borderRadius: 5, borderSkipped: false }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0, max: 55, ticks: { callback: (v) => v + "%" }, grid: { color: "#f3f4f6" } },
          x: { grid: { display: false } },
        },
      },
    });
    return () => chart.destroy();
  }, []);
  return <canvas id={id} ref={ref} />;
}

export default function EDA() {
  return (
    <div>
      <div className="metric-grid">
        {[
          ["7,043", "Total customers", "Rows in dataset"],
          ["20", "Features", "Predictor columns"],
          ["26.5%", "Churn rate", "1,869 churned"],
          ["32 mo", "Avg tenure", "Median 29 months"],
        ].map(([val, label, sub]) => (
          <div key={label} className="metric">
            <div className="metric-label">{label}</div>
            <div className="metric-value">{val}</div>
            <div className="metric-sub">{sub}</div>
          </div>
        ))}
      </div>

      <div className="chart-row">
        <div className="card">
          <div className="section-title">Churn by contract type</div>
          <div style={{ height: 200 }}>
            <BarChart id="c1" {...CHURN_BY_CONTRACT} />
          </div>
        </div>
        <div className="card">
          <div className="section-title">Churn by internet service</div>
          <div style={{ height: 200 }}>
            <BarChart id="c2" {...CHURN_BY_INTERNET} />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Top churn risk drivers</div>
        {DRIVERS.map(({ label, pct }) => (
          <div key={label} className="bar-row">
            <span className="bar-label" style={{ minWidth: 220 }}>{label}</span>
            <div className="bar-track" style={{ width: 220 }}>
              <div className="bar-fill" style={{ width: `${pct}%`, background: "#185FA5" }} />
            </div>
            <span className="bar-val">{pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
