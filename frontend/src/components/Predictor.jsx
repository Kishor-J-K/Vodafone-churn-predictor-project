import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "";

const DEFAULTS = {
  gender: "Male",
  senior_citizen: 0,
  partner: "No",
  dependents: "No",
  tenure: 6,
  phone_service: "Yes",
  multiple_lines: "No",
  internet_service: "Fiber optic",
  online_security: "No",
  online_backup: "No",
  device_protection: "No",
  tech_support: "No",
  streaming_tv: "No",
  streaming_movies: "No",
  contract: "Month-to-month",
  paperless_billing: "Yes",
  payment_method: "Electronic check",
  monthly_charges: 85,
  total_charges: 510,
};

const FIELDS = [
  { key: "contract",         label: "Contract type",     type: "select", opts: ["Month-to-month","One year","Two year"] },
  { key: "internet_service", label: "Internet service",  type: "select", opts: ["Fiber optic","DSL","No"] },
  { key: "tenure",           label: "Tenure (months)",   type: "number", min: 0, max: 72 },
  { key: "monthly_charges",  label: "Monthly charges ($)",type: "number", min: 18, max: 120, step: 0.5 },
  { key: "online_security",  label: "Online security",   type: "select", opts: ["No","Yes","No internet service"] },
  { key: "tech_support",     label: "Tech support",      type: "select", opts: ["No","Yes","No internet service"] },
  { key: "payment_method",   label: "Payment method",    type: "select", opts: ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"] },
  { key: "paperless_billing",label: "Paperless billing", type: "select", opts: ["Yes","No"] },
  { key: "senior_citizen",   label: "Senior citizen",    type: "select", opts: [0, 1], labels: ["No","Yes"] },
  { key: "dependents",       label: "Dependents",        type: "select", opts: ["No","Yes"] },
  { key: "gender",           label: "Gender",            type: "select", opts: ["Male","Female"] },
  { key: "partner",          label: "Partner",           type: "select", opts: ["Yes","No"] },
];

export default function Predictor() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (key, val) => setForm((f) => ({ ...f, [key]: val }));

  const predict = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const payload = {
        ...form,
        senior_citizen: parseInt(form.senior_citizen),
        tenure: parseInt(form.tenure),
        monthly_charges: parseFloat(form.monthly_charges),
        total_charges: parseFloat(form.monthly_charges) * parseInt(form.tenure) || parseFloat(form.monthly_charges),
      };
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const riskColor = result
    ? result.probability > 0.6 ? "#E24B4A" : result.probability > 0.35 ? "#EF9F27" : "#22c55e"
    : "#E24B4A";

  return (
    <div>
      <p style={{ fontSize: "0.85rem", color: "#6b7280", marginBottom: "1.5rem" }}>
        Fill in a customer profile and hit predict. The model runs on the trained XGBoost backend.
      </p>

      <div className="card">
        <div className="form-grid">
          {FIELDS.map(({ key, label, type, opts, labels, min, max, step }) => (
            <div key={key} className="form-group">
              <label>{label}</label>
              {type === "select" ? (
                <select value={form[key]} onChange={(e) => set(key, e.target.value)}>
                  {opts.map((o, i) => (
                    <option key={o} value={o}>{labels ? labels[i] : o}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={form[key]}
                  min={min}
                  max={max}
                  step={step || 1}
                  onChange={(e) => set(key, e.target.value)}
                />
              )}
            </div>
          ))}
        </div>

        <button className="predict-btn" onClick={predict} disabled={loading}>
          {loading ? "Predicting..." : "Predict churn risk →"}
        </button>

        {error && (
          <div className="result-box" style={{ background: "#FEF2F2", borderColor: "#FECACA" }}>
            <div className="result-title" style={{ color: "#991B1B" }}>Error</div>
            <div className="result-desc">{error}</div>
            <div className="result-desc" style={{ marginTop: 8 }}>
              Make sure the backend is running: <code>uvicorn app.main:app --reload</code>
            </div>
          </div>
        )}

        {result && (
          <div className={`result-box ${result.churn ? "churn" : "stay"}`}>
            <div className="result-title" style={{ color: result.churn ? "#991B1B" : "#166534" }}>
              {result.churn ? "Likely to churn" : "Likely to stay"} — {result.risk_level} risk
            </div>
            <div className="result-desc">
              Churn probability: {(result.probability * 100).toFixed(1)}%
            </div>

            <div className="risk-bar-wrap">
              <div className="risk-label">
                <span>Risk score</span>
                <span style={{ fontWeight: 600 }}>{(result.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="risk-track">
                <div className="risk-fill" style={{ width: `${result.probability * 100}%`, background: riskColor }} />
              </div>
            </div>

            <div className="factors">
              <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#6b7280", margin: "12px 0 6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Key factors
              </div>
              {result.top_factors.map((f) => (
                <div key={f} className="factor-item">
                  <div className="factor-dot" style={{ background: result.churn ? "#E24B4A" : "#22c55e" }} />
                  {f}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
