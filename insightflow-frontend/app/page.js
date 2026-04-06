"use client";

import { useState } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

const BASE_URL = "http://127.0.0.1:8080";

export default function Home() {
  const [output, setOutput] = useState(null);
  const [query, setQuery] = useState("");
  const [insights, setInsights] = useState(null);

  // ▶️ Run pipeline
  const runPipeline = async () => {
    const res = await axios.post(`${BASE_URL}/run-pipeline`);
    setOutput(res.data);
  };

  // ▶️ Get insights
  const getInsights = async () => {
    const res = await axios.get(`${BASE_URL}/get-insights`);
    setInsights(res.data);
    setOutput(res.data);
  };

  // ▶️ Query API
  const askQuery = async () => {
    const res = await axios.get(`${BASE_URL}/query?q=${query}`);
    setOutput(res.data);
  };

  // 📊 Prepare chart data
  const chartData =
    insights?.location_activity
      ? Object.entries(insights.location_activity).map(([key, value]) => ({
          location: key,
          count: value,
        }))
      : [];

  return (
    <div style={{ padding: 20 }}>
      <h1>🚀 InsightFlow AI Dashboard</h1>

      {/* Buttons */}
      <div style={{ marginBottom: 20 }}>
        <button onClick={runPipeline}>Run Pipeline</button>
        <button onClick={getInsights}>Get Insights</button>
      </div>

      {/* Query */}
      <div style={{ marginBottom: 20 }}>
        <h3>💬 Ask Question</h3>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Which locations have highest activity?"
          style={{ width: "300px" }}
        />
        <button onClick={askQuery}>Ask</button>
      </div>

      {/* Chart */}
      {chartData.length > 0 && (
        <div>
          <h3>📊 Activity by Location</h3>
          <BarChart width={600} height={300} data={chartData}>
            <XAxis dataKey="location" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" />
          </BarChart>
        </div>
      )}

      {/* Output */}
      <div style={{ marginTop: 20 }}>
        <h3>📦 Output</h3>
        <pre style={{ background: "#eee", padding: 10 }}>
          {JSON.stringify(output, null, 2)}
        </pre>
      </div>
    </div>
  );
}