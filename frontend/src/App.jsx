import { useState } from "react";
import EDA from "./components/EDA";
import Models from "./components/Models";
import Predictor from "./components/Predictor";
import "./App.css";

const tabs = ["EDA", "Model comparison", "Live predictor"];

export default function App() {
  const [active, setActive] = useState(0);

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <h1>Vodafone Churn Predictor</h1>
            <p className="subtitle">End-to-end ML · EDA · Model comparison · Live inference</p>
          </div>
          <span className="badge">XGBoost · AUC 0.92</span>
        </div>
      </header>

      <main className="main">
        <div className="tabs">
          {tabs.map((t, i) => (
            <button
              key={t}
              className={`tab ${active === i ? "active" : ""}`}
              onClick={() => setActive(i)}
            >
              {t}
            </button>
          ))}
        </div>

        {active === 0 && <EDA />}
        {active === 1 && <Models />}
        {active === 2 && <Predictor />}
      </main>
    </div>
  );
}
