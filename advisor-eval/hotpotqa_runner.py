"""HotpotQA fullwiki: Wikipedia-only agentic loop (search + passage extracts)."""

from __future__ import annotations

import json
import time
import urllib.parse
from typing import Any

from openai import OpenAI

from advisor import AdvisorAgent
from gaia_runner import (
    GaiaRunResult,
    _as_text,
    _build_working_memory_note,
    _estimate_gaia_confidence,
    _extract_json,
    _guidance_is_actionable,
    _http_get,
    _infer_answer_requirements,
    _is_hedged_answer,
    _normalise_query_key,
    _parse_next_action_from_guidance,
    _parse_tool_input,
    _short,
    _truncate_messages_for_advisor,
)
from policies import EscalationPolicy

HOTPOT_WIKI_SYSTEM_PROMPT = """\
You are an agent answering multi-hop questions using English Wikipedia only.

At each turn, output ONLY a valid JSON object with this schema:
{
  "thought": "brief reasoning",
  "request_advisor": false,
  "action": "tool" | "final",
  "tool_name": "wiki_search",
  "tool_input": "plain query string or JSON with query field",
  "final_answer": "short answer when action=final"
}

Tool:
- wiki_search: pass a focused English query. The tool returns search hits then
  passage text (article extracts) for the top pages. Use results to hop from
  entity A (bridge) to entity B, then answer.

Retrieval discipline:
- Parse the question for specific names (people, bases, stadiums, airports).
  Search those exact names first; do not substitute a different country or
  similarly named place (e.g. a US Army barracks vs an unrelated UK garrison).
- If the first hits look like the wrong entity, reformulate with more context
  from the question (unit name, state, country) before locking an answer.
- For comparison/bridge questions, gather evidence for each named item before
  action=final.

Final answer format (HotpotQA):
- Use the shortest string that fully answers the question: often one word
  (type, yes/no) or a place name, or digits for a year/number.
- For yes/no questions output "yes" or "no" (lowercase).
- For years, digits alone (e.g. 2000) are fine if the question asks for a year.
- If the question asks for a single category (e.g. "public" vs "private"),
  output that one word when the evidence supports it — not a long paraphrase.
- Do not add "The" or extra clauses unless the gold answer style requires it.

Rules:
- Do not invent facts not present in retrieved passages.
- If action is "tool", use tool_name="wiki_search" only.
- If action is "final", set final_answer and do not call tools.
- Never repeat the same wiki_search query; rephrase with new disambiguators.
- Set request_advisor=true when stuck, after tool errors, or when evidence conflicts.
"""


HOTPOT_WORKING_MEMORY_SUFFIX = (
    "\n\nHotpotQA hint: align search queries with named entities in the question; "
    "resolve the correct location/venue before answering. Prefer the minimal "
    "final_answer (one word or short span) that matches what the question asks."
)


def wiki_search_passages(
    parsed: Any,
    *,
    search_limit: int,
    top_k_pages: int,
    extract_chars_per_page: int,
    total_budget_chars: int,
) -> str:
    """Search English Wikipedia, then fetch plaintext extracts for top titles."""
    query = _as_text(parsed, key="query")
    if not query:
        raise ValueError("wiki_search requires a non-empty query string.")
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": min(search_limit, 50),
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    payload = json.loads(_http_get(url))
    items = payload.get("query", {}).get("search", [])
    if not items:
        return "No results."
    parts: list[str] = []
    budget_left = total_budget_chars
    for item in items[:top_k_pages]:
        title = (item.get("title") or "").strip()
        if not title or budget_left <= 0:
            break
        ex_limit = min(extract_chars_per_page, budget_left)
        ex_params = urllib.parse.urlencode({
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "exchars": ex_limit,
        })
        ex_url = f"https://en.wikipedia.org/w/api.php?{ex_params}"
        ex_payload = json.loads(_http_get(ex_url))
        pages = ex_payload.get("query", {}).get("pages", {})
        if not pages:
            continue
        page = next(iter(pages.values()))
        if page.get("missing"):
            continue
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        block = f"### {title}\n{extract}"
        if len(block) > budget_left:
            block = block[: budget_left - 1] + "…"
        parts.append(block)
        budget_left -= len(block)
    if not parts:
        return "No extractable passages for the search results."
    return "\n\n".join(parts)


def _build_hotpot_advisor_context_pack(
    *,
    trigger: str,
    question: str,
    tool_trace: list[dict[str, Any]],
    last_raw: str,
    last_action: dict[str, Any] | None,
    budget_used: int,
    budget_max: int,
    blocked_query_keys: set[str],
    evidence_snippets: list[str],
    candidate_answer: str | None,
    tool_error_streak: int,
    dead_end_count: int,
) -> str:
    lines: list[str] = [f"[ADVISOR CONSULT: trigger={trigger}]"]
    lines.append(f"Question: {question.strip()}")
    lines.append(f"Answer requirements: {_infer_answer_requirements(question)}")
    lines.append("Available tools: wiki_search only (open-domain Wikipedia retrieval).")
    lines.append(f"Tool budget: {budget_used}/{budget_max}")
    lines.append(
        f"Health signals: tool_error_streak={tool_error_streak}, dead_end_count={dead_end_count}"
    )
    if candidate_answer:
        lines.append(f"Current candidate answer: {_short(candidate_answer, 160)}")
    if evidence_snippets:
        lines.append("Recent evidence snippets:")
        for snippet in evidence_snippets[-3:]:
            lines.append(f"  - {snippet}")
    if blocked_query_keys:
        lines.append("Blocked duplicate queries:")
        for item in sorted(blocked_query_keys)[-5:]:
            lines.append(f"  - {item}")
    if tool_trace:
        lines.append(f"Tools used so far (n={len(tool_trace)}):")
        for idx, entry in enumerate(tool_trace[-10:], 1):
            tool = entry.get("tool", "?")
            inp = _short(str(entry.get("input", "")), 80)
            if entry.get("success"):
                preview = _short(str(entry.get("output_preview", "")), 100)
                lines.append(f"  {idx}. {tool}(\"{inp}\") -> ok: {preview}")
            else:
                err = _short(str(entry.get("error", "")), 100)
                lines.append(f"  {idx}. {tool}(\"{inp}\") -> error: {err}")
    else:
        lines.append("Tools used so far: none")
    if last_action is not None:
        act = str(last_action.get("action", "?"))
        if act == "final":
            lines.append(
                f"Executor's last step: action=final, final_answer="
                f"\"{_short(str(last_action.get('final_answer','')), 120)}\""
            )
        else:
            lines.append(
                f"Executor's last step: action=tool, "
                f"tool_name={last_action.get('tool_name','?')}, "
                f"tool_input=\"{_short(str(last_action.get('tool_input','')), 120)}\""
            )
    elif last_raw:
        lines.append(f"Executor's last raw output: {_short(last_raw, 200)}")
    lines.append("Give DIAGNOSIS + NEXT + AVOID now.")
    return "\n".join(lines)


def run_hotpot_wiki_agentic(
    *,
    question: str,
    executor_model: str,
    advisor: AdvisorAgent | None,
    policy: EscalationPolicy,
    policy_name: str,
    temperature: float,
    seed: int,
    max_steps: int,
    max_tool_calls: int,
    max_advisor_calls: int = 2,
    retrieve_config: dict[str, Any] | None = None,
) -> GaiaRunResult:
    """Agentic loop with Wikipedia search+extract only and optional advisor."""
    rcfg = retrieve_config or {}
    search_limit = int(rcfg.get("search_limit", 8))
    top_k_pages = int(rcfg.get("top_k_pages", 3))
    extract_chars_per_page = int(rcfg.get("extract_chars_per_page", 2500))
    total_budget_chars = int(rcfg.get("total_budget_chars", 12000))

    client = OpenAI()
    result = GaiaRunResult()
    had_error = False
    post_error_recovered = False

    messages: list[dict[str, str]] = [
        {"role": "system", "content": HOTPOT_WIKI_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    recent_query_keys: list[str] = []
    blocked_query_keys: set[str] = set()
    evidence_snippets: list[str] = []
    candidate_answer: str | None = None
    tool_error_streak = 0
    cooldown_until = -1
    last_action_dict: dict[str, Any] | None = None
    force_answer_used = False
    late_rescue_used = False
    no_progress_advisor_rounds = 0
    blocked_hosts: dict[str, int] = {}

    for step_idx in range(max_steps):
        step_lat = 0.0
        t0 = time.perf_counter()
        try:
            working_memory_note = _build_working_memory_note(
                evidence_snippets=evidence_snippets,
                blocked_query_keys=blocked_query_keys,
                blocked_hosts=blocked_hosts,
                candidate_answer=candidate_answer,
            ) + HOTPOT_WORKING_MEMORY_SUFFIX
            response = client.chat.completions.create(
                model=executor_model,
                messages=messages + [{"role": "user", "content": working_memory_note}],
                temperature=temperature,
                seed=seed,
                max_completion_tokens=512,
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            result.total_exec_latency += latency
            result.tool_trace.append({
                "step": step_idx,
                "tool": "executor_api",
                "input": "",
                "success": False,
                "error": f"{type(exc).__name__}: {exc}"[:400],
            })
            result.step_latencies.append(latency)
            result.dead_end_count += 1
            result.recovery_success = False
            return result
        latency = time.perf_counter() - t0
        step_lat += latency

        usage = response.usage
        result.total_exec_latency += latency
        result.total_exec_prompt += usage.prompt_tokens if usage else 0
        result.total_exec_completion += usage.completion_tokens if usage else 0

        raw_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw_text})

        action = _extract_json(raw_text)
        parse_error = action is None
        tool_error = False
        done = False
        wants_advisor = False
        answer: str | None = None
        duplicate_query = False

        if parse_error:
            result.dead_end_count += 1
            tool_error = True
            had_error = True
            tool_error_streak += 1
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Return only a valid JSON object following the schema."
                ),
            })
            result.tool_trace.append({
                "step": step_idx,
                "tool": "json_parse",
                "input": "",
                "success": False,
                "error": "invalid_json",
                "output_preview": (raw_text or "")[:400],
            })
        else:
            action_type = str(action.get("action", "")).strip().lower()
            if action_type == "wiki_search":
                action["tool_name"] = "wiki_search"
                action_type = "tool"
            wants_advisor = bool(action.get("request_advisor", False))
            last_action_dict = {
                "action": action_type,
                "tool_name": action.get("tool_name", ""),
                "tool_input": action.get("tool_input", ""),
                "final_answer": action.get("final_answer", ""),
            }

            if action_type == "final":
                done = True
                answer = str(action.get("final_answer", "")).strip() or None
                candidate_answer = answer
                tool_error_streak = 0
            elif action_type == "tool":
                if result.tool_calls >= max_tool_calls:
                    tool_error = True
                    had_error = True
                    result.dead_end_count += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool budget exceeded ({max_tool_calls}). "
                            "Decide with available evidence and return action=final."
                        ),
                    })
                else:
                    tool_name = str(action.get("tool_name", "")).strip()
                    tool_input = str(action.get("tool_input", "")).strip()
                    parsed_tool_input = _parse_tool_input(tool_input)
                    qkey = _normalise_query_key(tool_name, tool_input)
                    if qkey in recent_query_keys:
                        duplicate_query = True
                    if qkey in blocked_query_keys:
                        duplicate_query = True
                        result.repeated_query_violations += 1
                        tool_error = True
                        had_error = True
                        tool_error_streak += 1
                        result.dead_end_count += 1
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[TOOL ERROR] name={tool_name}\ninput={tool_input}\n"
                                "error=duplicate_or_blocked_query\n"
                                "Use a materially different query."
                            ),
                        })
                    if qkey not in blocked_query_keys:
                        recent_query_keys.append(qkey)
                        if len(recent_query_keys) > 6:
                            recent_query_keys = recent_query_keys[-6:]
                        result.tool_calls += 1
                        success = True
                        output = ""
                        error = ""
                        try:
                            if tool_name == "wiki_search":
                                output = wiki_search_passages(
                                    parsed_tool_input,
                                    search_limit=search_limit,
                                    top_k_pages=top_k_pages,
                                    extract_chars_per_page=extract_chars_per_page,
                                    total_budget_chars=total_budget_chars,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown tool '{tool_name}'. Only wiki_search is available."
                                )
                        except Exception as exc:  # noqa: BLE001
                            success = False
                            error = str(exc)
                            output = ""
                    else:
                        success = False
                        output = ""
                        error = "duplicate_or_blocked_query"

                    if success:
                        if had_error:
                            post_error_recovered = True
                        tool_error_streak = 0
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[TOOL RESULT] name={tool_name}\n"
                                f"input={tool_input}\n"
                                f"output={output}"
                            ),
                        })
                        evidence_snippets.append(f"{tool_name}: {_short(output, 110)}")
                    else:
                        tool_error = True
                        had_error = True
                        tool_error_streak += 1
                        result.tool_errors += 1
                        result.dead_end_count += 1
                        blocked_query_keys.add(qkey)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[TOOL ERROR] name={tool_name}\n"
                                f"input={tool_input}\n"
                                f"error={error}\n"
                                "Adjust your wiki_search query."
                            ),
                        })

                    result.tool_trace.append({
                        "step": step_idx,
                        "tool": tool_name,
                        "input": tool_input,
                        "success": success,
                        "error": error,
                        "output_preview": output[:240],
                    })
            else:
                tool_error = True
                had_error = True
                result.dead_end_count += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "Invalid action type. Use action='tool' with wiki_search "
                        "or action='final' in valid JSON."
                    ),
                })
                result.tool_trace.append({
                    "step": step_idx,
                    "tool": "action_validation",
                    "input": str(action.get("action", "")),
                    "success": False,
                    "error": "invalid_action",
                })

        hedged_final = done and _is_hedged_answer(answer)
        budget_fraction = (
            result.tool_calls / max_tool_calls if max_tool_calls > 0 else 0.0
        )
        confidence = _estimate_gaia_confidence(
            done=done,
            hedged_final=hedged_final,
            parse_error=parse_error,
            tool_error=tool_error,
            duplicate_query=duplicate_query,
            tool_error_streak=tool_error_streak,
            dead_end_count=result.dead_end_count,
            evidence_count=len(evidence_snippets),
        )
        policy_result: dict[str, Any] = {
            "text": raw_text,
            "answer": answer,
            "wants_advisor": wants_advisor,
            "done": done,
            "tool_error": tool_error,
            "parse_error": parse_error,
            "duplicate_query": duplicate_query,
            "hedged_final": hedged_final,
            "tool_error_streak": tool_error_streak,
            "budget_fraction": budget_fraction,
            "confidence": confidence,
        }
        policy_state: dict[str, Any] = {
            "step": step_idx,
            "messages": messages,
            "dead_end_count": result.dead_end_count,
            "tool_errors": result.tool_errors,
            "tool_error_streak": tool_error_streak,
            "duplicate_query_streak": sum(
                1 for k in recent_query_keys[-2:] if recent_query_keys.count(k) >= 2
            ),
            "budget_fraction": budget_fraction,
            "hedged_final": hedged_final,
        }

        stuck_triggers: list[str] = []
        if tool_error_streak >= 2:
            stuck_triggers.append("two_tool_errors")
        if duplicate_query:
            stuck_triggers.append("duplicate_query")
        if budget_fraction >= 0.75 and not done:
            stuck_triggers.append("budget_near_limit")
        if hedged_final:
            stuck_triggers.append("hedged_final_answer")
        if done and len(evidence_snippets) < 2:
            stuck_triggers.append("low_evidence_final")
        if done and not wants_advisor and (
            hedged_final or len(evidence_snippets) < 2 or result.tool_errors > 0
        ):
            stuck_triggers.append("auto_low_confidence_final")
        if wants_advisor:
            stuck_triggers.append("model_requested")
        if parse_error:
            stuck_triggers.append("parse_error")

        policy_triggered = advisor is not None and policy.should_escalate(
            step_idx, policy_result, policy_state
        )
        stuck_triggered = advisor is not None and bool(stuck_triggers)

        advisor_cap_hit = (
            max_advisor_calls is not None
            and result.advisor_calls >= max_advisor_calls
        )
        late_rescue = done and (hedged_final or len(evidence_snippets) < 2)
        tool_budget_left = max_tool_calls - result.tool_calls
        budget_guard_block = tool_budget_left < 3 and not hedged_final

        suppress_random = (
            policy_name == "random_prob"
            and (step_idx < 2 or tool_budget_left <= 3)
            and not stuck_triggered
            and not hedged_final
        )
        if suppress_random:
            policy_triggered = False

        if late_rescue and late_rescue_used:
            late_rescue = False
        in_cooldown = step_idx <= cooldown_until and not late_rescue
        blocked = advisor_cap_hit or (budget_guard_block and not late_rescue)
        should_escalate = (
            (policy_triggered or stuck_triggered)
            and not in_cooldown
            and not blocked
        )

        if should_escalate and advisor is not None:
            if stuck_triggers:
                trigger_name = stuck_triggers[0]
            else:
                trigger_name = policy_name or "policy"
            context_pack = _build_hotpot_advisor_context_pack(
                trigger=trigger_name,
                question=question,
                tool_trace=result.tool_trace,
                last_raw=raw_text,
                last_action=last_action_dict,
                budget_used=result.tool_calls,
                budget_max=max_tool_calls,
                blocked_query_keys=blocked_query_keys,
                evidence_snippets=evidence_snippets,
                candidate_answer=candidate_answer,
                tool_error_streak=tool_error_streak,
                dead_end_count=result.dead_end_count,
            )
            advisor_messages = _truncate_messages_for_advisor(messages) + [
                {"role": "user", "content": context_pack}
            ]
            try:
                guidance, adv_stats = advisor.advise(advisor_messages)
            except Exception as exc:  # noqa: BLE001
                result.tool_trace.append({
                    "step": step_idx,
                    "tool": "advisor_api",
                    "input": trigger_name,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                })
                guidance = None
                adv_stats = None
            if guidance is not None and adv_stats is not None:
                if not _guidance_is_actionable(guidance, result.tool_trace, blocked_hosts):
                    guidance = None
                    adv_stats = None
            if guidance is not None and adv_stats is not None:
                result.total_adv_latency += adv_stats.latency_s
                result.total_adv_prompt += adv_stats.prompt_tokens
                result.total_adv_completion += adv_stats.completion_tokens
                result.advisor_calls += 1
                if late_rescue:
                    late_rescue_used = True
                if had_error:
                    result.advisor_calls_after_error += 1
                next_tool_hint, next_input_hint = _parse_next_action_from_guidance(guidance)
                result.advisor_guidance.append({
                    "step": step_idx,
                    "after_error": had_error,
                    "trigger": trigger_name,
                    "guidance": guidance,
                    "next_tool_hint": next_tool_hint,
                    "next_input_hint": next_input_hint,
                    "followed_advice": None,
                })
                step_lat += adv_stats.latency_s
                messages = advisor.integrate_advice(
                    messages,
                    guidance,
                    format_hint=(
                        "Apply this guidance. On your NEXT turn, execute step "
                        "1 of the advisor's NEXT list exactly, unless it is "
                        "clearly impossible. Do NOT repeat any query listed "
                        "in AVOID or any query you already tried. Respond "
                        "with the same JSON schema as before."
                    ),
                )
                if hedged_final:
                    done = False
                cooldown_until = step_idx + 2
                no_progress_advisor_rounds = 0
            else:
                no_progress_advisor_rounds += 1
                if no_progress_advisor_rounds >= 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Stop requesting advisor repeatedly without progress. "
                            "Use remaining evidence to produce your best final answer now."
                        ),
                    })

        if (
            result.advisor_guidance
            and result.advisor_guidance[-1].get("followed_advice") is None
            and last_action_dict is not None
            and result.advisor_guidance[-1].get("step") == step_idx - 1
        ):
            hint = result.advisor_guidance[-1].get("next_tool_hint")
            if hint:
                actual_tool = (
                    last_action_dict.get("tool_name", "")
                    if last_action_dict.get("action") == "tool"
                    else None
                )
                followed = actual_tool == hint
                result.advisor_guidance[-1]["followed_advice"] = followed
                result.advisor_first_step_total += 1
                if followed:
                    result.advisor_followed_first_step_count += 1
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You did not follow the advisor's first NEXT step. "
                            "On this turn, execute that first step unless impossible; "
                            "if impossible, explain briefly in thought and reformulate "
                            "wiki_search."
                        ),
                    })

        result.step_latencies.append(step_lat)
        if done and not should_escalate:
            result.prediction = answer
            break

        if policy_name.startswith("self_eval") or policy_name.startswith("failure_or_conf_t"):
            result.confidence_scores.append(confidence)

    if (
        (result.prediction is None or _is_hedged_answer(result.prediction))
        and not force_answer_used
    ):
        force_answer_used = True
        messages.append({
            "role": "user",
            "content": (
                "You must commit to a final answer now using the evidence "
                "gathered so far. Output ONLY a valid JSON object with "
                "action='final' and a non-empty final_answer. Use the briefest "
                "correct span (often one word or a place/year). Do not call wiki_search."
            ),
        })
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=executor_model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_completion_tokens=256,
            )
            latency = time.perf_counter() - t0
            result.total_exec_latency += latency
            usage = resp.usage
            result.total_exec_prompt += usage.prompt_tokens if usage else 0
            result.total_exec_completion += usage.completion_tokens if usage else 0
            forced_text = resp.choices[0].message.content or ""
            forced = _extract_json(forced_text)
            if forced and str(forced.get("action", "")).lower() == "final":
                forced_answer = str(forced.get("final_answer", "")).strip()
                if forced_answer and not _is_hedged_answer(forced_answer):
                    result.prediction = forced_answer
        except Exception:  # noqa: BLE001
            pass

    result.recovery_success = had_error and post_error_recovered
    return result
