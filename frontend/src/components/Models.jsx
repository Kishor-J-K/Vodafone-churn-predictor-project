import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

const MODELS = [
  { name: "Logistic Regression", f1: 0.72, auc: 0.84, prec: 0.69, rec: 0.76, best: false },
  { name: "Decision Tree",       f1: 0.74, auc: 0.82, prec: 0.71, rec: 0.77, best: false },
  { name: "Random Forest",       f1: 0.79, auc: 0.88, prec: 0.78, rec: 0.80, best: false },
  { name: "XGBoost",             f1: 0.83, auc: 0.92, prec: 0.82, rec: 0.84, best: true  },
];

const FEATURES = [
  { name: "Contract type", imp: 0.31 },
  { name: "Tenure", imp: 0.22 },
  { name: "Monthly charges", imp: 0.18 },
  { name: "Internet service", imp: 0.11 },
  { name: "Online security", imp: 0.07 },
  { name: "Tech support", imp: 0.05 },
  { name: "Payment method", imp: 0.03 },
  { name: "Senior citizen", imp: 0.01 },
];

export default function Models() {
  const chartRef = useRef(null);

  useEffect(() => {
    const ctx = chartRef.current.getContext("2d");
    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: FEATURES.map((f) => f.name),
        datasets: [{
          data: FEATURES.map((f) => f.imp),
          backgroundColor: "#185FA5",
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { callback: (v) => v.toFixed(2) }, grid: { color: "#f3f4f6" } },
          y: { grid: { display: false } },
        },
      },
    });
    return () => chart.destroy();
  }, []);

  return (
    <div>
      <p style={{ fontSize: "0.85rem", color: "#6b7280", marginBottom: "1.5rem" }}>
        All models trained on 80/20 stratified split with SMOTE oversampling for class imbalance.
        Metrics computed on held-out test set.
      </p>

      <div className="model-grid">
        {MODELS.map((m) => (
          <div key={m.name} className={`model-card ${m.best ? "best" : ""}`}>
            {m.best && <div className="best-badge">Best model</div>}
            <div className="model-name">{m.name}</div>
            {[["F1", m.f1], ["AUC", m.auc], ["Precision", m.prec], ["Recall", m.rec]].map(([l, v]) => (
              <div key={l} className="bar-row">
                <span className="bar-label" style={{ width: 60, minWidth: 60 }}>{l}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${v * 100}%`, background: m.best ? "#185FA5" : "#9ca3af" }} />
                </div>
                <span className="bar-val">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="card">
        <div className="section-title">Feature importance — XGBoost</div>
        <div style={{ height: 280 }}>
          <canvas ref={chartRef} />
        </div>
      </div>
    </div>
  );
}
