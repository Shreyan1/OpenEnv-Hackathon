from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schemas import ConversationTurn, Episode, MemoryType


@dataclass(frozen=True)
class PhrasedOption:
    label: str
    keyword: str
    text: str
    phrasing_index: int


PreferenceSpec = Tuple[str, str, Tuple[str, ...]]
ConstraintSpec = Tuple[str, Tuple[str, ...]]
ProjectSpec = Tuple[str, str, Tuple[str, ...]]


_PREFERENCES: List[PreferenceSpec] = [
    (
        "ClickHouse",
        "clickhouse",
        (
            "We're running ClickHouse on the analytics side, so that's the stack to target.",
            "I'd go with ClickHouse here; it handles our query volume better.",
            "The data team standardized on ClickHouse last quarter.",
            "ClickHouse is what I know best for this workflow, so let's stick with it.",
        ),
    ),
    (
        "PostgreSQL",
        "postgresql",
        (
            "Let's keep this on PostgreSQL; it's what the rest of the platform already uses.",
            "We're a PostgreSQL shop for this service, so optimize around that.",
            "PostgreSQL is the default here unless there's a really strong reason not to use it.",
            "Infra wants this landing on PostgreSQL, so plan around that.",
        ),
    ),
    (
        "SQLite",
        "sqlite",
        (
            "For this prototype I'd rather keep it on SQLite and avoid extra moving parts.",
            "SQLite is good enough for the first pass here, so let's keep it simple.",
            "I'd start with SQLite for now; we can swap it out later if needed.",
            "Let's stay with SQLite while we're still validating the workflow.",
        ),
    ),
    (
        "Python",
        "python",
        (
            "I'd prefer Python for this one; the team can move faster with it.",
            "Let's write it in Python unless there's a performance blocker.",
            "Python is still the path of least resistance for this service.",
            "We're strongest in Python on this team, so optimize for that.",
        ),
    ),
    (
        "FastAPI",
        "fastapi",
        (
            "If we're building an API here, let's use FastAPI.",
            "FastAPI would fit this service better than rolling something lower-level.",
            "I'd rather keep the HTTP layer in FastAPI; it's what our examples already use.",
            "Let's standardize on FastAPI for the service boundary.",
        ),
    ),
    (
        "Redis",
        "redis",
        (
            "Redis is the right fit if we need something lightweight and fast here.",
            "I'd rather back this with Redis than introduce a heavier dependency.",
            "Redis is still the simplest option for the caching layer.",
            "Let's use Redis for the transient state in this flow.",
        ),
    ),
    (
        "MongoDB",
        "mongodb",
        (
            "The document-heavy shape here points me toward MongoDB.",
            "I'd use MongoDB for this instead of forcing it into a relational model.",
            "MongoDB is probably the cleaner fit for the payloads we're discussing.",
            "If we want schema flexibility, MongoDB is the obvious choice.",
        ),
    ),
    (
        "TypeScript",
        "typescript",
        (
            "I'd rather keep the implementation in TypeScript so the frontend folks can contribute too.",
            "Let's do this in TypeScript; the team already has the tooling wired up.",
            "TypeScript is the safer default here given how many people touch this code.",
            "I'd stick with TypeScript for this service so the types travel cleanly across the stack.",
        ),
    ),
    (
        "DuckDB",
        "duckdb",
        (
            "DuckDB is probably the cleanest option for the local analytics bit.",
            "I'd reach for DuckDB here before spinning up anything heavier.",
            "DuckDB should cover this workload without extra infra overhead.",
            "Let's use DuckDB for the embedded analytics path.",
        ),
    ),
    (
        "gRPC",
        "grpc",
        (
            "If we're talking service-to-service calls, I'd prefer gRPC.",
            "Let's keep the internal API on gRPC; it's a better fit for this path.",
            "I'd rather expose this over gRPC than a chatty REST interface.",
            "gRPC makes more sense here given the internal traffic patterns.",
        ),
    ),
]

_CONSTRAINTS: List[ConstraintSpec] = [
    (
        "bullet points",
        (
            "Can you keep it in bullet points? Easier to scan.",
            "Bullet-pointed list please; I'm pasting this into a doc.",
            "Format it as bullets so I can drop it straight into Notion.",
            "Please answer in bullet points; the team reads these quickly.",
        ),
    ),
    (
        "numbered list",
        (
            "Make it a numbered list so I can reference each step.",
            "Please structure the response as a numbered list.",
            "I'd like the answer in a numbered list for review comments.",
            "Can you number the sections so they're easy to call out?",
        ),
    ),
    (
        "five sentences",
        (
            "Keep the answer under five sentences if you can.",
            "Please stay within about five sentences; this is going to exec staff.",
            "I only need a short answer here, around five sentences max.",
            "Let's keep this to roughly five sentences so it stays readable.",
        ),
    ),
    (
        "concise",
        (
            "Keep it concise; I just need the essentials.",
            "Please make the response concise, not a long essay.",
            "A short answer is better here; I only need the key points.",
            "Keep it tight; this is just going in a quick handoff note.",
        ),
    ),
    (
        "valid json",
        (
            "I need the output as valid JSON because another tool will parse it.",
            "Please return valid JSON only; no extra narration around it.",
            "Make the response valid JSON so I can feed it into a script.",
            "Can you keep the output to valid JSON? It's for an automated check.",
        ),
    ),
    (
        "code examples",
        (
            "Please include a code example so the implementation path is obvious.",
            "I want the answer to include at least one code example.",
            "Can you show an example snippet in the response?",
            "Add a code example so the team has something concrete to start from.",
        ),
    ),
    (
        "type annotations",
        (
            "Include type annotations in any function examples.",
            "If you show code, please keep the type annotations in place.",
            "Add type annotations to the examples; I don't want to infer the shapes.",
            "Keep the code samples typed so they're easier to review.",
        ),
    ),
    (
        "snake_case",
        (
            "Use snake_case for identifiers in the example.",
            "Please stick to snake_case naming throughout.",
            "Can you keep the identifiers in snake_case? That's our style here.",
            "Use snake_case names so the example matches the codebase.",
        ),
    ),
]

_PROJECT_FACTS: List[ProjectSpec] = [
    (
        "memory budget",
        "memory budget",
        (
            "The whole point of this project is learning under a fixed memory budget.",
            "We care about operating under a hard memory budget, not unlimited context.",
            "The benchmark only matters if the policy respects the memory budget.",
            "Please keep in mind that memory budget pressure is a core part of the task.",
        ),
    ),
    (
        "deterministic evaluation",
        "deterministic",
        (
            "We need the evaluation to stay deterministic so runs are comparable.",
            "Deterministic grading matters here; we can't depend on a flaky judge.",
            "The benchmark has to stay deterministic or the scores won't mean much.",
            "Make sure the evaluation path remains deterministic; that's a hard requirement.",
        ),
    ),
    (
        "production deployment",
        "production",
        (
            "This isn't just a toy; we want something we could plausibly deploy.",
            "We're aiming at a production-shaped setup, not a demo-only benchmark.",
            "The environment needs to feel production-relevant, otherwise the score won't matter.",
            "Please treat this like something we'd actually ship, not just a class project.",
        ),
    ),
    (
        "latency under 100 milliseconds",
        "100 milliseconds",
        (
            "The latency target is still under 100 milliseconds end to end.",
            "We have to stay under a 100 millisecond latency budget.",
            "Keep the 100 millisecond SLA in mind while you answer.",
            "The service target is sub-100 milliseconds, so avoid heavy suggestions.",
        ),
    ),
    (
        "multi-tenant workloads",
        "multi-tenant",
        (
            "This needs to work in a multi-tenant setup, not just for one user.",
            "We're designing for multi-tenant workloads, so isolation matters.",
            "Keep the multi-tenant requirement in mind here.",
            "The project has to support multiple tenants cleanly.",
        ),
    ),
    (
        "GitHub Actions CI",
        "github actions",
        (
            "Our CI still runs in GitHub Actions, so keep examples compatible with that.",
            "This all lands in GitHub Actions in the end, so keep the workflow practical.",
            "Anything you suggest should fit a normal GitHub Actions pipeline.",
            "The build path goes through GitHub Actions, which shapes what we can automate.",
        ),
    ),
]

_DISTRACTORS: Tuple[str, ...] = (
    "Sorry for the slow reply; I've been in back-to-back meetings all afternoon.",
    "The deploy finally cleared and everything looks green now.",
    "I'll be out on Thursday, so if anything is urgent please flag it early.",
    "I still need to clean up a couple of old tickets after this.",
    "Not urgent, but I also need to figure out why my laptop fan won't calm down.",
    "I can open a separate ticket for the unrelated setup question if that's easier.",
    "The release notes draft is almost done; I just need to polish it.",
    "I'm still catching up after the incident review this morning.",
    "We can talk about onboarding docs later; that's separate from this thread.",
    "Just noting that the staging deploy used the usual checklist and passed.",
)

_CONFABULATION_TEMPLATES: Tuple[str, ...] = (
    "My colleague keeps pushing for {tech}, but that's not actually the direction I want here.",
    "Hypothetically, if we were on {tech}, how would this look? I'm not saying we should switch.",
    "The old team used {tech}, but that's not the setup we're targeting now.",
    "I've heard people recommend {tech}, although it's not relevant for this request.",
    "Someone on another team mentioned {tech}; treat that as background chatter, not a requirement.",
    "A vendor deck kept bringing up {tech}, but I'm not asking you to use it.",
    "If this were a {tech} shop the answer might change, but it isn't.",
    "There was a proposal around {tech} a while back, but we didn't adopt it.",
)

_CORRECTION_TEMPLATES: Tuple[str, ...] = (
    "Actually, scratch the {old}; the infra update says we're on {new} now.",
    "Update from the team: forget {old}, we should use {new}.",
    "Correction: swap out {old} and plan around {new} instead.",
    "Change of plan: don't optimize for {old}; optimize for {new}.",
)

_RECALL_CHECK_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "preference": (
        "Before I send this around, what stack did I say we were using?",
        "Quick check: what was my preferred technology again?",
        "Remind me which tool I said we should target here.",
        "What platform choice did I land on earlier in the thread?",
    ),
    "constraint": (
        "What formatting rule did I ask for again?",
        "Quick reminder: how did I want the answer formatted?",
        "Can you recall the output constraint I mentioned earlier?",
        "What response format did I ask you to stick to?",
    ),
    "project_info": (
        "What project context did I mention that should shape the answer?",
        "Before we wrap, remind me of the broader project requirement I called out.",
        "What background detail about the project did I mention earlier?",
        "Can you recall the project constraint I flagged in this thread?",
    ),
}

_FINAL_QUERY_TEMPLATES: Tuple[str, ...] = (
    "Can you write the final response now? Use {preference} as the stack choice and follow the '{constraint}' instruction I gave earlier.",
    "Please draft the final answer using {preference} and make sure it respects the '{constraint}' requirement.",
    "Give me the final response for this thread. It should reflect {preference} and honor the '{constraint}' formatting note.",
    "Wrap this up into a final answer that uses {preference} and follows the '{constraint}' instruction from earlier.",
)

_EASY_FINAL_QUERY_TEMPLATES: Tuple[str, ...] = (
    "Can you write the final answer now using the preference I mentioned for the stack: {preference}?",
    "Please send the final response and make sure it reflects my earlier stack choice: {preference}.",
    "Wrap this into a final answer that uses {preference}, since that's what I asked for earlier.",
)


def _sample_from_pool(rng: random.Random, pool: Sequence[str]) -> Tuple[str, int]:
    index = rng.randrange(len(pool))
    return pool[index], index


def sample_preference(rng: random.Random, *, exclude_keywords: Sequence[str] = ()) -> PhrasedOption:
    blocked = {keyword.lower() for keyword in exclude_keywords}
    options = [spec for spec in _PREFERENCES if spec[1].lower() not in blocked]
    if not options:
        options = list(_PREFERENCES)
    label, keyword, phrases = rng.choice(options)
    text, phrasing_index = _sample_from_pool(rng, phrases)
    return PhrasedOption(label=label, keyword=keyword, text=text, phrasing_index=phrasing_index)


def sample_constraint(rng: random.Random) -> PhrasedOption:
    keyword, phrases = rng.choice(_CONSTRAINTS)
    text, phrasing_index = _sample_from_pool(rng, phrases)
    return PhrasedOption(label=keyword, keyword=keyword, text=text, phrasing_index=phrasing_index)


def sample_project_fact(rng: random.Random) -> PhrasedOption:
    label, keyword, phrases = rng.choice(_PROJECT_FACTS)
    text, phrasing_index = _sample_from_pool(rng, phrases)
    return PhrasedOption(label=label, keyword=keyword, text=text, phrasing_index=phrasing_index)


def make_preference_turn(turn_id: int, option: PhrasedOption) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        text=option.text,
        kind="preference",
        memory_type=MemoryType.PREFERENCE,
        metadata={"keyword": option.keyword, "phrasing_index": option.phrasing_index, "label": option.label},
    )


def make_constraint_turn(turn_id: int, option: PhrasedOption) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        text=option.text,
        kind="constraint",
        memory_type=MemoryType.CONSTRAINT,
        metadata={"keyword": option.keyword, "phrasing_index": option.phrasing_index, "label": option.label},
    )


def make_project_turn(turn_id: int, option: PhrasedOption) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        text=option.text,
        kind="project_info",
        memory_type=MemoryType.PROJECT_INFO,
        metadata={"keyword": option.keyword, "phrasing_index": option.phrasing_index, "label": option.label},
    )


def make_distractor_turn(turn_id: int, rng: random.Random) -> ConversationTurn:
    text, phrasing_index = _sample_from_pool(rng, _DISTRACTORS)
    return ConversationTurn(
        turn_id=turn_id,
        text=text,
        kind="distractor",
        metadata={"phrasing_index": phrasing_index},
    )


def make_confabulation_turn(
    turn_id: int,
    rng: random.Random,
    *,
    blocked_keywords: Sequence[str] = (),
) -> ConversationTurn:
    option = sample_preference(rng, exclude_keywords=blocked_keywords)
    template, phrasing_index = _sample_from_pool(rng, _CONFABULATION_TEMPLATES)
    return ConversationTurn(
        turn_id=turn_id,
        text=template.format(tech=option.label),
        kind="confabulation",
        metadata={
            "keyword": option.keyword,
            "technology": option.label,
            "phrasing_index": phrasing_index,
        },
    )


def make_correction_turn(turn_id: int, rng: random.Random, previous: PhrasedOption) -> Tuple[ConversationTurn, PhrasedOption]:
    replacement = sample_preference(rng, exclude_keywords=(previous.keyword,))
    template, phrasing_index = _sample_from_pool(rng, _CORRECTION_TEMPLATES)
    return (
        ConversationTurn(
            turn_id=turn_id,
            text=template.format(old=previous.label, new=replacement.label),
            kind="correction",
            memory_type=MemoryType.PREFERENCE,
            metadata={
                "keyword": replacement.keyword,
                "old_keyword": previous.keyword,
                "old_label": previous.label,
                "label": replacement.label,
                "phrasing_index": phrasing_index,
                "correction_of": "preference",
            },
        ),
        replacement,
    )


def make_recall_check_turn(
    turn_id: int,
    rng: random.Random,
    *,
    target_kind: str,
    target_keyword: str,
) -> ConversationTurn:
    templates = _RECALL_CHECK_TEMPLATES[target_kind]
    text, phrasing_index = _sample_from_pool(rng, templates)
    memory_type = {
        "preference": MemoryType.PREFERENCE,
        "constraint": MemoryType.CONSTRAINT,
        "project_info": MemoryType.PROJECT_INFO,
    }[target_kind]
    return ConversationTurn(
        turn_id=turn_id,
        text=text,
        kind="recall_check",
        memory_type=memory_type,
        metadata={"keyword": target_keyword, "target_kind": target_kind, "phrasing_index": phrasing_index},
    )


def make_final_query_turn(
    turn_id: int,
    rng: random.Random,
    *,
    preference: PhrasedOption,
    constraint: PhrasedOption,
    project: PhrasedOption | None = None,
) -> ConversationTurn:
    template, phrasing_index = _sample_from_pool(rng, _FINAL_QUERY_TEMPLATES)
    text = template.format(preference=preference.label, constraint=constraint.keyword)
    if project is not None:
        text = f"{text} Also keep the project context in mind: {project.label}."
    return ConversationTurn(
        turn_id=turn_id,
        text=text,
        kind="final_query",
        metadata={"phrasing_index": phrasing_index},
    )


def make_easy_final_query_turn(turn_id: int, rng: random.Random, *, preference: PhrasedOption) -> ConversationTurn:
    template, phrasing_index = _sample_from_pool(rng, _EASY_FINAL_QUERY_TEMPLATES)
    return ConversationTurn(
        turn_id=turn_id,
        text=template.format(preference=preference.label),
        kind="final_query",
        metadata={"phrasing_index": phrasing_index},
    )


def build_episode_metadata(
    *,
    turns: Sequence[ConversationTurn],
    memory_budget: int,
    required_memory_types: Sequence[MemoryType],
    required_keywords: Sequence[str],
    latest_preference_keyword: str = "",
    latest_constraint_keyword: str = "",
    latest_project_keyword: str = "",
    expose_turn_kind: bool = True,
    decay_rate: float = 0.0,
    task_id: str = "synthetic_default",
) -> Dict[str, object]:
    return {
        "required_memory_types": [memory_type.value for memory_type in required_memory_types],
        "required_keywords": [keyword for keyword in required_keywords if keyword],
        "final_query": turns[-1].text if turns else "",
        "latest_preference_keyword": latest_preference_keyword,
        "latest_constraint_keyword": latest_constraint_keyword,
        "latest_project_keyword": latest_project_keyword,
        "turn_count": len(turns),
        "memory_budget": memory_budget,
        "task_id": task_id,
        "expose_turn_kind": expose_turn_kind,
        "decay_rate": decay_rate,
    }


@dataclass
class SyntheticEpisodeGenerator:
    memory_budget: int = 200
    min_turns: int = 6
    max_turns: int = 8

    def generate(self, seed: int | None = None) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"episode_{episode_seed}_{rng.randint(1000, 9999)}"

        preference = sample_preference(rng)
        constraint = sample_constraint(rng)
        project = sample_project_fact(rng)

        turns: List[ConversationTurn] = [
            make_preference_turn(0, preference),
            make_distractor_turn(1, rng),
            make_constraint_turn(2, constraint),
        ]

        if rng.random() > 0.5:
            correction_turn, preference = make_correction_turn(3, rng, preference)
            turns.append(correction_turn)
        else:
            turns.append(make_project_turn(3, project))

        turns.append(
            make_recall_check_turn(
                4,
                rng,
                target_kind="preference",
                target_keyword=preference.keyword,
            )
        )
        turns.append(make_distractor_turn(5, rng))

        if rng.random() > 0.7 and len(turns) < self.max_turns - 1:
            turns.append(
                make_confabulation_turn(
                    len(turns),
                    rng,
                    blocked_keywords=(preference.keyword,),
                )
            )

        turns.append(make_final_query_turn(len(turns), rng, preference=preference, constraint=constraint))
        turns = turns[: self.max_turns]
        turns = [
            ConversationTurn(
                turn_id=index,
                text=turn.text,
                kind=turn.kind,
                memory_type=turn.memory_type,
                tags=turn.tags,
                metadata=turn.metadata,
            )
            for index, turn in enumerate(turns)
        ]

        metadata = build_episode_metadata(
            turns=turns,
            memory_budget=self.memory_budget,
            required_memory_types=(MemoryType.PREFERENCE, MemoryType.CONSTRAINT),
            required_keywords=(preference.keyword, constraint.keyword),
            latest_preference_keyword=preference.keyword,
            latest_constraint_keyword=constraint.keyword,
            expose_turn_kind=True,
            decay_rate=0.0,
            task_id="synthetic_default",
        )

        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.memory_budget,
            metadata=metadata,
        )
