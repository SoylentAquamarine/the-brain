from datetime import datetime, timezone, timedelta


def recent_provider_health(data: dict, window_hours=24):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    out = {}

    for key, entries in data.items():
        ok = 0
        total = 0
        lats = []

        for e in entries:
            ts = datetime.fromisoformat(e["ts"])
            if ts < cutoff:
                continue

            total += 1
            ok += e["ok"]
            lats.append(e["latency"])

        if total == 0:
            continue

        provider = key.split(":")[0]

        if provider not in out:
            out[provider] = {"ok": 0, "total": 0, "latencies": []}

        out[provider]["ok"] += ok
        out[provider]["total"] += total
        out[provider]["latencies"].extend(lats)

    final = {}

    for p, d in out.items():
        final[p] = {
            "uptime": round(d["ok"] / d["total"], 3),
            "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"]), 1) if d["latencies"] else 0,
            "sample_count": d["total"]
        }

    return final