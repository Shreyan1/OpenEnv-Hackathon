from __future__ import annotations

import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory_management_agent import (  # noqa: E402
    Action,
    ActionType,
    EmbeddingRetrievalAgent,
    KeywordRetrievalAgent,
    MemoryManagementEnv,
    MemoryStore,
    PreferenceOnlyAgent,
    RuleBasedMemoryAgent,
    SyntheticEpisodeGenerator,
)
from memory_management_agent.evaluation import evaluate_split, hidden_eval_seeds, run_episode  # noqa: E402
from memory_management_agent.training import build_policy_prompt, parse_action_block, summarize_rollouts  # noqa: E402
from memory_management_agent.training import TrainingConfig, run_training_experiment  # noqa: E402
from memory_management_agent.analysis import analyze_rollouts, memory_evolution_text  # noqa: E402
from memory_management_agent.review import render_failure_cases, render_full_review  # noqa: E402
from memory_management_agent.training import collect_rollouts  # noqa: E402


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
        self.assertGreaterEqual(episode_result.reward, -1.0)


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


if __name__ == "__main__":
    unittest.main()
