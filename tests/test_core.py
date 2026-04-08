from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock
import importlib
import types

from fastapi.testclient import TestClient


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory_management_agent import (  # noqa: E402
    Action,
    ActionType,
    EmbeddingRetrievalAgent,
    Grader,
    KeywordRetrievalAgent,
    MemoryManagementEnv,
    MemoryStore,
    PreferenceOnlyAgent,
    RuleBasedMemoryAgent,
    SyntheticEpisodeGenerator,
    TASK_HARD,
    TASK_MEDIUM,
    generator_for_task,
    normalize_task_score,
)
from memory_management_agent.evaluation import evaluate_split, hidden_eval_seeds, run_episode  # noqa: E402
from memory_management_agent.training import build_policy_prompt, parse_action_block, summarize_rollouts  # noqa: E402
from memory_management_agent.training import TrainingConfig, run_training_experiment  # noqa: E402
from memory_management_agent.analysis import analyze_rollouts, memory_evolution_text  # noqa: E402
from memory_management_agent.review import render_failure_cases, render_full_review  # noqa: E402
from memory_management_agent.training import collect_rollouts  # noqa: E402


def _import_inference_with_fake_openai() -> types.ModuleType:
    fake_openai = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **_kwargs: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ACTION: ignore\nTEXT:\nIDS:"))]
                    )
                )
            )

    fake_openai.OpenAI = DummyOpenAI  # type: ignore[attr-defined]
    sys.modules.pop("inference", None)
    with mock.patch.dict(sys.modules, {"openai": fake_openai}):
        return importlib.import_module("inference")


def _import_run_llm_agent_with_fake_openenv() -> types.ModuleType:
    generic_client_module = types.ModuleType("openenv.core.generic_client")
    client_types_module = types.ModuleType("openenv.core.client_types")
    llm_client_module = types.ModuleType("openenv.core.llm_client")

    class DummyGenericEnvClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def _parse_result(self, payload: dict) -> object:
            return types.SimpleNamespace(
                observation=payload.get("observation"),
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

    generic_client_module.GenericEnvClient = DummyGenericEnvClient  # type: ignore[attr-defined]
    client_types_module.StepResult = object  # type: ignore[attr-defined]
    llm_client_module.create_llm_client = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    sys.modules.pop("run_llm_agent", None)
    with mock.patch.dict(
        sys.modules,
        {
            "openenv": types.ModuleType("openenv"),
            "openenv.core": types.ModuleType("openenv.core"),
            "openenv.core.generic_client": generic_client_module,
            "openenv.core.client_types": client_types_module,
            "openenv.core.llm_client": llm_client_module,
        },
    ):
        return importlib.import_module("run_llm_agent")


class EpisodeGenerationTests(unittest.TestCase):
    def test_generator_is_deterministic_for_seed(self) -> None:
        generator = SyntheticEpisodeGenerator(memory_budget=100)
        left = generator.generate(seed=7)
        right = generator.generate(seed=7)

        self.assertEqual(left.to_dict(), right.to_dict())
        self.assertGreaterEqual(len(left.turns), 6)
        self.assertEqual(left.turns[-1].kind, "final_query")


class MemoryStoreTests(unittest.TestCase):
    def test_store_respects_budget_and_queries(self) -> None:
        store = MemoryStore(budget_tokens=5)
        item, inserted, evicted = store.add("Use PostgreSQL", memory_type="preference", turn_index=1)  # type: ignore[arg-type]
        self.assertTrue(inserted)
        self.assertFalse(evicted)
        self.assertEqual(item.text, "Use PostgreSQL")

        store.add(
            "Keep answers concise",
            memory_type="constraint",  # type: ignore[arg-type]
            turn_index=2,
        )
        self.assertLessEqual(store.total_tokens, 5)
        results = store.query("What database should I use?", k=1)
        self.assertEqual(len(results), 1)

    def test_decay_reduces_stale_utility(self) -> None:
        store = MemoryStore(budget_tokens=50, decay_rate=0.1, decay_window=1)
        item, _, _ = store.add("Use PostgreSQL", memory_type="preference", turn_index=0, utility_score=0.8)  # type: ignore[arg-type]
        store.apply_decay(current_turn=1)
        self.assertAlmostEqual(store.snapshot()[0].utility_score, item.utility_score)
        store.apply_decay(current_turn=3)
        self.assertLess(store.snapshot()[0].utility_score, item.utility_score)


class EnvironmentTests(unittest.TestCase):
    def test_environment_runs_to_completion(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        observation = env.reset(seed=11)
        self.assertEqual(observation.step_number, 0)

        done = False
        steps = 0
        while not done:
            current = observation.current_turn_kind
            if current == "final_query":
                action = Action.answer("Here is the answer with the required memory.")
            elif current in {"preference", "constraint", "correction", "project_info"}:
                action = Action.store(observation.current_user_message)
            else:
                action = Action.ignore()
            result = env.step(action)
            observation = result.observation if result.observation is not None else observation
            done = result.done
            steps += 1

        self.assertGreaterEqual(steps, 5)
        episode_result = env.build_episode_result()
        self.assertIsNotNone(episode_result.metrics)
        self.assertGreater(episode_result.reward, 0.0)
        self.assertLess(episode_result.reward, 1.0)

    def test_normalize_task_score_uses_strict_open_interval(self) -> None:
        self.assertEqual(normalize_task_score(-1.0), 0.0001)
        self.assertEqual(normalize_task_score(0.0), 0.0001)
        self.assertEqual(normalize_task_score(0.25), 0.25)
        self.assertEqual(normalize_task_score(1.0), 0.9999)
        self.assertEqual(normalize_task_score(5.0), 0.9999)

    def test_hidden_turn_kind_observation_masks_recent_context(self) -> None:
        env = MemoryManagementEnv(generator=generator_for_task(TASK_MEDIUM))
        observation = env.reset(seed=11)
        self.assertEqual(observation.current_turn_kind, "unknown")

        next_observation = env.step(Action.ignore()).observation
        self.assertIsNotNone(next_observation)
        assert next_observation is not None
        self.assertEqual(next_observation.current_turn_kind, "unknown")
        self.assertTrue(all(turn.kind == "unknown" for turn in next_observation.recent_conversation))
        self.assertTrue(all(turn.memory_type is None for turn in next_observation.recent_conversation))

    def test_hard_task_contains_confabulation_and_two_corrections(self) -> None:
        episode = generator_for_task(TASK_HARD).generate(seed=17)
        kinds = [turn.kind for turn in episode.turns]
        self.assertEqual(kinds.count("correction"), 2)
        self.assertIn("confabulation", kinds)
        self.assertIn("project_info", kinds)
        self.assertEqual(episode.metadata["required_memory_types"], ["preference", "constraint", "project_info"])


class BaselineAgentTests(unittest.TestCase):
    def test_rule_based_agent_completes_episode(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        result = run_episode(RuleBasedMemoryAgent(), env, seed=3)
        self.assertIsInstance(result.reward, float)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_preference_only_agent_completes_episode(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        result = run_episode(PreferenceOnlyAgent(), env, seed=5)
        self.assertIsInstance(result.reward, float)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_keyword_retrieval_agent_completes_episode(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        result = run_episode(KeywordRetrievalAgent(), env, seed=9)
        self.assertIsInstance(result.reward, float)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_embedding_retrieval_agent_completes_episode(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        result = run_episode(EmbeddingRetrievalAgent(), env, seed=13)
        self.assertIsInstance(result.reward, float)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_hidden_evaluation_runs_visible_and_hidden_sets(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        report = evaluate_split(
            RuleBasedMemoryAgent(),
            env,
            visible_seeds=(1, 2),
            hidden_seeds=hidden_eval_seeds(count=2, start=5000),
        )
        self.assertEqual(report.agent_name, "RuleBasedMemoryAgent")
        self.assertEqual(len(report.visible.results), 2)
        self.assertEqual(len(report.hidden.results), 2)
        self.assertIsInstance(report.generalization_gap, float)


class TrainingScaffoldTests(unittest.TestCase):
    def test_prompt_builder_includes_memory_and_format(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        observation = env.reset(seed=21)
        prompt = build_policy_prompt(observation)

        self.assertIn("You are a memory management policy.", prompt.observation_prompt)
        self.assertIn("Memory bank:", prompt.observation_prompt)
        self.assertIn("ACTION:", prompt.action_format)

    def test_grader_scores_constraint_adherence(self) -> None:
        episode = generator_for_task(TASK_MEDIUM).generate(seed=2)
        grader = Grader()
        bad = grader.score_episode(episode, (), "PostgreSQL in prose", ())
        constraint_keyword = episode.metadata["latest_constraint_keyword"]
        good_answer_by_constraint = {
            "bullet points": "- Use PostgreSQL.\n- Keep the implementation aligned with the latest preference.",
            "numbered list": "1. Use PostgreSQL.\n2. Keep the implementation aligned with the latest preference.",
            "five sentences": (
                "Use PostgreSQL for the stack. "
                "Keep the implementation simple. "
                "Reflect the corrected preference. "
                "Preserve the existing workflow. "
                "Ship the smallest viable change."
            ),
            "concise": "Use PostgreSQL and keep the response short.",
            "valid json": '{"database":"postgresql","note":"use the corrected preference"}',
            "code examples": "```python\nprint('postgresql')\n```",
            "type annotations": "def choose_db() -> str:\n    return 'postgresql'",
            "snake_case": "use_postgresql_for_this_service",
        }
        good_answer = good_answer_by_constraint[str(constraint_keyword)]
        good = grader.score_episode(episode, (), good_answer, ())
        self.assertLess(bad.constraint_adherence, good.constraint_adherence)

    def test_action_parser_understands_answer_blocks(self) -> None:
        action = parse_action_block(
            "\n".join(
                [
                    "ACTION: ANSWER",
                    "TEXT: Here is the answer.",
                    "IDS: mem_0001, mem_0002",
                ]
            )
        )
        self.assertEqual(action.type, ActionType.ANSWER)
        self.assertEqual(action.text, "Here is the answer.")

    def test_rollout_summary_handles_empty_input(self) -> None:
        summary = summarize_rollouts(())
        self.assertEqual(summary["average_reward"], 0.0)
        self.assertEqual(summary["average_success"], 0.0)

    def test_training_experiment_writes_artifacts(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_training_experiment(
                RuleBasedMemoryAgent(),
                env,
                train_seeds=(1, 2),
                visible_eval_seeds=(3,),
                hidden_eval_seeds=(5001,),
                output_dir=tmpdir,
                config=TrainingConfig(run_name="unit-test"),
            )
            self.assertEqual(report.config.run_name, "unit-test")
            self.assertTrue(os.path.exists(report.manifest_path))
            self.assertTrue(os.path.exists(report.checkpoint_path))
            self.assertTrue(os.path.exists(report.benchmark_path))
            self.assertTrue(os.path.exists(report.train_rollout_path))

    def test_analysis_generates_summary_and_memory_evolution(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        rollouts = collect_rollouts(RuleBasedMemoryAgent(), env, seeds=(1, 2))
        analysis = analyze_rollouts(rollouts)
        self.assertEqual(analysis.total_episodes, 2)
        self.assertIn("memory evolution", memory_evolution_text(rollouts[0]).lower())
        self.assertGreaterEqual(sum(analysis.action_counts.values()), 2)

    def test_review_renderers_produce_text(self) -> None:
        env = MemoryManagementEnv(memory_budget=120)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_training_experiment(
                RuleBasedMemoryAgent(),
                env,
                train_seeds=(1,),
                visible_eval_seeds=(2,),
                hidden_eval_seeds=(5002,),
                output_dir=tmpdir,
                config=TrainingConfig(run_name="review-test"),
            )
            text = render_full_review(report)
            self.assertIn("Run: review-test", text)
            self.assertIn("Visible reward", text)
            self.assertIn("Generalization gap", text)
            analysis = analyze_rollouts(collect_rollouts(RuleBasedMemoryAgent(), env, seeds=(3,)))
            self.assertIn("Failure cases", render_failure_cases(analysis))


class LoggingTests(unittest.TestCase):
    def test_inference_parser_maps_invalid_turn_kind_to_safe_action(self) -> None:
        inference = _import_inference_with_fake_openai()
        parsed = inference._parse_action("ACTION: recall_check\nTEXT:\nIDS:")
        self.assertEqual(parsed["type"], "retrieve")

    def test_inference_main_emits_start_step_end_logs(self) -> None:
        inference = _import_inference_with_fake_openai()
        stdout = StringIO()
        stderr = StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "test-token",
                    "API_BASE_URL": "https://example.invalid/v1",
                    "MODEL_NAME": "fake-model",
                },
                clear=False,
            ),
            mock.patch.object(inference, "HF_TOKEN", "test-token"),
            mock.patch.object(inference, "API_BASE_URL", "https://example.invalid/v1"),
            mock.patch.object(inference, "MODEL_NAME", "fake-model"),
            mock.patch.object(inference, "_run_episode", return_value=(0.5, 1)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            inference.main()

        out = stdout.getvalue()
        # Validator-facing structured lines (must be on stdout).
        self.assertEqual(out.count("[START] task="), 3)
        self.assertEqual(out.count("[STEP] step="), 9)
        self.assertEqual(out.count("[END] task="), 3)
        self.assertIn("score=", out)
        self.assertIn("steps=", out)

        err = stderr.getvalue()
        # Detailed logs stay on stderr.
        self.assertIn("[START] event=\"inference_run\"", err)
        self.assertIn("[START] event=\"task_run\"", err)
        self.assertIn("[STEP] event=\"seed_result\"", err)
        self.assertIn("[END] event=\"task_run\"", err)
        self.assertIn("[END] event=\"inference_run\"", err)

    def test_server_routes_emit_start_step_end_logs(self) -> None:
        from server.app import GraderRequest, ResetRequest, StepRequest, grader, reset, step

        stderr = StringIO()
        with redirect_stderr(stderr):
            reset_response = reset(ResetRequest(task_id="easy_preference_recall", seed=42))
            session_id = reset_response.session_id

            step_response = step(StepRequest(session_id=session_id, action={"type": "ignore"}))
            self.assertFalse(step_response.done)

            grader_response = grader(GraderRequest(session_id=session_id))
            self.assertEqual(grader_response.task_id, "easy_preference_recall")
            self.assertGreater(grader_response.score, 0.0)
            self.assertLess(grader_response.score, 1.0)

        output = stderr.getvalue()
        self.assertIn("[START] event=\"http_reset\"", output)
        self.assertIn("[STEP] event=\"http_reset_session_created\"", output)
        self.assertIn("[END] event=\"http_reset\"", output)
        self.assertIn("[START] event=\"http_step\"", output)
        self.assertIn("[STEP] event=\"http_step_action_received\"", output)
        self.assertIn("[END] event=\"http_step\"", output)
        self.assertIn("[START] event=\"http_grader\"", output)
        self.assertIn("[STEP] event=\"http_grader_result\"", output)
        self.assertIn("[END] event=\"http_grader\"", output)

    def test_http_reset_accepts_empty_post_body(self) -> None:
        from server.app import app

        client = TestClient(app)
        response = client.post("/reset")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("session_id", payload)
        self.assertEqual(payload["task_id"], "easy_preference_recall")
        self.assertIn("observation", payload)

    def test_http_grader_clamps_boundary_scores_into_open_interval(self) -> None:
        from server.app import app

        client = TestClient(app)
        reset_response = client.post("/reset", json={"task_id": "easy_preference_recall", "seed": 1})
        session_id = reset_response.json()["session_id"]

        with mock.patch(
            "server.app.MemoryManagementEnv.build_episode_result",
            return_value=types.SimpleNamespace(
                reward=1.0,
                metrics=types.SimpleNamespace(to_dict=lambda: {}),
                final_answer="done",
            ),
        ):
            response = client.post("/grader", json={"session_id": session_id})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 0.9999)

    def test_run_llm_agent_requires_terminal_score(self) -> None:
        run_llm_agent = _import_run_llm_agent_with_fake_openenv()
        client = run_llm_agent._MemoryEnvClient()

        with self.assertRaisesRegex(ValueError, "Missing terminal task score"):
            client._parse_result({"done": True, "reward": 0.2})

    def test_run_llm_agent_normalizes_boundary_score(self) -> None:
        run_llm_agent = _import_run_llm_agent_with_fake_openenv()

        self.assertEqual(run_llm_agent._require_task_score(0.0), 0.0001)
        self.assertEqual(run_llm_agent._require_task_score(1.0), 0.9999)

    def test_http_step_terminal_reward_uses_strict_final_score(self) -> None:
        from server.app import app

        client = TestClient(app)
        reset_response = client.post("/reset", json={"task_id": "easy_preference_recall", "seed": 1})
        session_id = reset_response.json()["session_id"]

        response = client.post(
            "/step",
            json={"session_id": session_id, "action": {"type": "answer", "text": "answering too early"}},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["done"])
        self.assertGreater(payload["reward"], 0.0)
        self.assertLess(payload["reward"], 1.0)

    def test_websocket_terminal_reward_and_score_use_strict_interval(self) -> None:
        from server.app import app

        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "reset", "data": {"task_id": "easy_preference_recall", "seed": 1}})
            reset_payload = websocket.receive_json()
            self.assertEqual(reset_payload["type"], "observation")

            websocket.send_json({"type": "step", "data": {"type": "answer", "text": "answering too early"}})
            step_payload = websocket.receive_json()

        self.assertEqual(step_payload["type"], "observation")
        data = step_payload["data"]
        self.assertTrue(data["done"])
        self.assertGreater(data["reward"], 0.0)
        self.assertLess(data["reward"], 1.0)
        self.assertGreater(data["score"], 0.0)
        self.assertLess(data["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
