"""Build the remaining d15 capture requests from the original fixed fixtures."""

import json
from hashlib import sha256
from pathlib import Path

OUT = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one template marker: {old!r}")
    return text.replace(old, new)


def main():
    target = OUT / "d15_resume.html"
    if target.exists():
        raise FileExistsError(target)
    source = OUT / "d15_fixtures.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    cohorts = []
    segments = []
    for p, start in ((.005, 8484), (.007, 0)):
        cohort, = [c for c in original["surface"]["cohorts"] if c["p"] == p]
        if len(cohort["shots"]) != 10_000:
            raise ValueError("Original cohort must contain exactly 10000 requests")
        resumed = dict(cohort, shots=cohort["shots"][start:], shot_start=start, original_shots=10_000)
        cohorts.append(resumed)
        segments.append({"p": p, "seed": cohort["seed"], "shot_start": start,
                         "shot_stop": 10_000, "planned": len(resumed["shots"]),
                         "request_tokens_preserved": True})
    payload = dict(original, surface=dict(original["surface"], cohorts=cohorts,
                                         joint_fixture="d15_joint.npz", resume=True))
    encoded = json.dumps(payload, separators=(",", ":"))
    page = (OUT / "d15.html").read_text(encoding="utf-8")
    marker = '<script id="embedded-fixtures" type="application/json">'
    left = page.index(marker)+len(marker)
    right = page.index("</script>", left)
    page = page[:left]+encoded+page[right:]
    page = replace_once(page, "const surfaceTimeoutMs = 250;", "const surfaceTimeoutMs = 1000;")
    page = replace_once(page,
        "const cap = Math.min(...config.cohorts.map(cohort => cohort.shots.length));",
        "const cap = Math.max(...config.cohorts.map(cohort => cohort.shots.length));")
    page = replace_once(page, "current = {mode, image_tag: config.image_tag, p: cohort.p",
        "current = {mode, image_tag: config.image_tag, shot_start: cohort.shot_start ?? 0, original_shots: cohort.original_shots ?? cohort.shots.length, p: cohort.p")
    page = replace_once(page, "telescope-d15-deck-v2-2026-11-22.json", "telescope-d15-deck-v2-resume.json")
    old_open = """  async function openLink() {
    await port.open({baudRate: config.baud_rate, dataBits: 8, stopBits: 1, parity: 'none', flowControl: 'none'});
    opened = true;
    reader = port.readable.getReader();
    writer = port.writable.getWriter();
    await new Promise(resolve => setTimeout(resolve, baudSettleMs));
  }"""
    new_open = """  async function openLink(recovery = null) {
    for (let attempt = 1; attempt <= 5; attempt += 1) {
      try {
        await port.open({baudRate: config.baud_rate, dataBits: 8, stopBits: 1, parity: 'none', flowControl: 'none'});
        opened = true;
        break;
      } catch (error) {
        const detail = {...(recovery ?? {}), stage: 'port_open', attempt,
          kind: error.kind ?? 'serial_open', message: error.message};
        if (current) current.recoveries.push(detail);
        else (measurement.transport_recoveries ??= []).push(detail);
        if (attempt === 5) throw error;
        await new Promise(resolve => setTimeout(resolve, 250));
      }
    }
    reader = port.readable.getReader();
    writer = port.writable.getWriter();
    await new Promise(resolve => setTimeout(resolve, baudSettleMs));
  }"""
    page = replace_once(page, old_open, new_open)
    page = replace_once(page, "        await closeLink();\n        await openLink();",
        "        await closeLink();\n        await new Promise(resolve => setTimeout(resolve, 250));\n        await openLink({shot: shotIndex + 1, warmup, request_retry: retry + 1});")
    target.write_text(page, encoding="utf-8")
    (OUT / "d15_resume_fixtures.json").write_text(encoded, encoding="utf-8")
    plan = {
        "original_fixtures": source.name, "original_fixtures_sha256": sha256(source.read_bytes()).hexdigest(),
        "original_joint": "d15_joint.npz", "image_tag": original["surface"]["image_tag"],
        "completed_before_resume": 28_484, "remaining_validated_shots": 11_516,
        "segments": segments, "page": target.name, "fixtures": "d15_resume_fixtures.json",
        "capture_filename": "telescope-d15-deck-v2-resume.json", "surface_timeout_ms": 1000,
        "reopen_attempts": 5, "reopen_delay_ms": 250,
        "merge": "Append each resume run at shot_start in the matching original p/seed cohort; request bytes and original tokens are unchanged",
    }
    (OUT / "d15_resume_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    print(target.as_uri(), flush=True)
    print(json.dumps(segments), flush=True)


if __name__ == "__main__":
    main()
