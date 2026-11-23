"use client";

import type { AdminActivityPoint } from "@/types/admin";

type Metric = "registrations" | "active_users" | "recommendations";

interface ActivityBarChartProps {
  title: string;
  definition: string;
  points: AdminActivityPoint[];
  metric: Metric;
  color: string;
}

function shortDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function ActivityBarChart({
  title,
  definition,
  points,
  metric,
  color,
}: ActivityBarChartProps) {
  const values = points.map((point) => point[metric]);
  const max = Math.max(1, ...values);
  const total = values.reduce((sum, value) => sum + value, 0);
  const height = 170;
  const top = 14;
  const bottom = 24;
  const usable = height - top - bottom;
  const step = 12;
  const width = Math.max(420, points.length * step);
  const barWidth = Math.max(4, step - 3);
  const first = points[0]?.date;
  const last = points[points.length - 1]?.date;

  return (
    <section
      aria-labelledby={`chart-${metric}`}
      className="rounded-2xl border border-white/10 bg-white/[0.02] p-5"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2
            id={`chart-${metric}`}
            className="text-sm font-medium text-white"
          >
            {title}
          </h2>
          <p className="mt-1 text-[11px] leading-relaxed text-white/40">
            {definition}
          </p>
        </div>
        <p className="font-mono text-[11px] text-white/50">
          {total.toLocaleString("en-US")} total
        </p>
      </div>

      <div className="mt-4 overflow-x-auto">
        <svg
          role="img"
          aria-label={`${title}: ${total} total over ${points.length} days`}
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          className="block"
        >
          <line
            x1={0}
            y1={height - bottom}
            x2={width}
            y2={height - bottom}
            stroke="rgba(255,255,255,0.12)"
            strokeWidth={1}
          />
          <line
            x1={0}
            y1={top + usable / 2}
            x2={width}
            y2={top + usable / 2}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={1}
          />
          {points.map((point, index) => {
            const value = point[metric];
            const barHeight = (value / max) * usable;
            const x = index * step + (step - barWidth) / 2;
            const y = top + usable - barHeight;
            return (
              <rect
                key={point.date}
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(0, barHeight)}
                rx={1.5}
                fill={color}
                opacity={value === 0 ? 0.15 : 0.9}
              >
                <title>
                  {point.date}: {value.toLocaleString("en-US")}
                </title>
              </rect>
            );
          })}
        </svg>
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-white/30">
        <span>{first ? shortDate(first) : ""}</span>
        <span>{last ? shortDate(last) : ""}</span>
      </div>

      <table className="sr-only">
        <caption>{title}</caption>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {points.map((point) => (
            <tr key={point.date}>
              <td>{point.date}</td>
              <td>{point[metric]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
