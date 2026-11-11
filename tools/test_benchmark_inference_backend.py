from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import benchmark_inference_backend as benchmark_module
from tools import backend_runtime
from tools import inference_project_runtime


class PriorityMetricTests(unittest.TestCase):
    def test_content_end_is_separate_from_stream_end(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"b"}}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":4}}\n',
                    b'data: [DONE]\n',
                ])

        with (
            mock.patch.object(benchmark_module.urllib.request, "urlopen", return_value=Response()),
            mock.patch.object(benchmark_module.time, "perf_counter", side_effect=[1.0, 1.02, 1.07, 1.08]),
        ):
            row = benchmark_module._run_request("http://localhost", "m", {"prompt": "p"}, 64, 0, 5)
        self.assertEqual(row["first_content_offset_ms"], 20.0)
        self.assertEqual(row["last_content_offset_ms"], 70.0)
        self.assertEqual(row["completion_offset_ms"], 80.0)
        self.assertEqual(row["content_tpot_proxy_ms"], 16.667)
        self.assertEqual(row["tpot_ms"], 20.0)  # Preserve the shared legacy metric.
        self.assertTrue(row["stream_done_received"])

    def test_empty_and_single_token_streams_do_not_invent_content_intervals(self) -> None:
        for content, tokens in (("", 0), ("x", 1)):
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.__iter__.return_value = iter([
                ("data: " + json.dumps({"choices": [{"delta": {"content": content},
                 "finish_reason": "stop"}], "usage": {"completion_tokens": tokens}}) + "\n").encode(),
                b'data: [DONE]\n',
            ])
            with mock.patch.object(benchmark_module.urllib.request, "urlopen", return_value=response):
                row = benchmark_module._run_request("http://localhost", "m", {"prompt": "p"}, 64, 0, 5)
            self.assertIsNone(row["content_tpot_proxy_ms"])
            if not content:
                self.assertIsNone(row["first_content_offset_ms"])
                self.assertIsNone(row["last_content_offset_ms"])

    def test_request_sends_chat_template_settings_and_records_finish_reason(self) -> None:
        class MockResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"Answer: 42"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n',
                ])

        with mock.patch.object(benchmark_module.urllib.request, "urlopen", return_value=MockResponse()) as urlopen:
            result = benchmark_module._run_request(
                "http://127.0.0.1:8000", "test-model",
                {
                    "prompt": "Add 35 and 7", "request_id": "r1", "expected_answer": "42",
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                16, 0.0, 5.0,
            )
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["expected_answer"], "42")
        self.assertEqual(result["output_text"], "Answer: 42")

    def test_task_quality_separates_answer_from_completion_and_text(self) -> None:
        rows = [
            {
                "ok": True,
                "request_id": "q42",
                "expected_answer": "42",
                "output_text": "Answer: 42",
                "finish_reason": "stop",
            },
            {
                "ok": True,
                "request_id": "q49",
                "expected_answer": "49",
                "output_text": "Answer: 49",
                "finish_reason": "length",
            },
            {
                "ok": True,
                "request_id": "q26",
                "expected_answer": "26",
                "output_text": "The total is 26.",
                "finish_reason": "stop",
            },
        ]
        metrics = benchmark_module._attach_task_quality(rows)
        self.assertEqual(metrics["expected_answer_accuracy"], 0.6667)
        self.assertEqual(metrics["valid_answer_format_rate"], 0.6667)
        self.assertEqual(metrics["response_completion_rate"], 0.6667)
        self.assertEqual(metrics["complete_correct_rate"], 0.3333)
        self.assertEqual(metrics["truncated_response_rate"], 0.3333)
        self.assertEqual(metrics["unique_answer_tasks"], 3)
        self.assertEqual(metrics["answer_repeat_consistency"], 0.6667)

    def test_request_specs_preserve_answer_and_chat_template_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            workload = Path(temporary_dir) / "workload.jsonl"
            workload.write_text(
                '{"request_id":"r1","prompt":"Add 35 and 7","expected_answer":42,'
                '"chat_template_kwargs":{"enable_thinking":false}}\n',
                encoding="utf-8",
            )
            specs = benchmark_module._read_request_specs(str(workload), "unused", 1)
        self.assertEqual(specs[0]["expected_answer"], "42")
        self.assertEqual(specs[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_first_token_score_uses_first_token_rank(self) -> None:
        results = [
            {"request_id": "high-1", "priority": 0, "first_token_rank": 1},
            {"request_id": "low-1", "priority": 5, "first_token_rank": 2},
            {"request_id": "high-2", "priority": 0, "first_token_rank": 3},
            {"request_id": "low-2", "priority": 5, "first_token_rank": 4},
        ]
        self.assertEqual(
            benchmark_module._priority_pair_score(results, "first_token_rank"),
            0.75,
        )

    def test_repeated_report_marks_request_result_scope(self) -> None:
        reports = [
            {
                "workload": {"repeats": 1},
                "metrics": {"priority_first_token_order_score": score},
                "results": [{"request_id": f"run-{index}"}],
            }
            for index, score in enumerate((0.5, 1.0), start=1)
        ]
        aggregate = benchmark_module.aggregate_repeated_reports(reports)
        self.assertEqual(aggregate["metrics"]["priority_first_token_order_score"], 0.75)
        self.assertEqual(aggregate["metrics_aggregation"], "mean_of_run_metrics")
        self.assertIn("first_run", aggregate["results_scope"])
        self.assertEqual(len(aggregate["runs"]), 2)

    def test_synchronized_start_requires_full_concurrency(self) -> None:
        args = SimpleNamespace(
            workload=None,
            prompt="test",
            num_prompts=2,
            synchronize_start=True,
            concurrency=1,
        )
        with mock.patch.object(
            benchmark_module,
            "_read_request_specs",
            return_value=[{"prompt": "a"}, {"prompt": "b"}],
        ):
            with self.assertRaisesRegex(ValueError, "concurrency"):
                benchmark_module.benchmark(args)


class BackendQueuePressureTests(unittest.TestCase):
    def test_optional_runtime_forwards_queue_pressure(self) -> None:
        with (
            mock.patch.object(backend_runtime, "resolve_model", return_value="/models/test"),
            mock.patch.object(
                backend_runtime,
                "start_vllm",
                return_value=(mock.Mock(), mock.Mock(), 8123, "bfloat16"),
            ) as start,
        ):
            result = inference_project_runtime.start_optional_vllm(
                model_id="test/model",
                max_num_seqs=2,
                scheduling_policy="priority",
            )
        self.assertEqual(result[2:], (8123, "bfloat16", "/models/test"))
        self.assertEqual(start.call_args.kwargs["max_num_seqs"], 2)
        self.assertEqual(start.call_args.kwargs["scheduling_policy"], "priority")

    def test_start_vllm_passes_max_num_seqs(self) -> None:
        process = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                mock.patch.object(backend_runtime, "find_free_port", return_value=8123),
                mock.patch.object(backend_runtime, "choose_dtype", return_value="bfloat16"),
                mock.patch.object(backend_runtime, "resolve_vllm_command", return_value="vllm"),
                mock.patch.object(backend_runtime, "wait_until_ready"),
                mock.patch.object(backend_runtime.subprocess, "Popen", return_value=process) as popen,
            ):
                _, log_handle, _, _ = backend_runtime.start_vllm(
                    "/models/test",
                    max_num_seqs=2,
                    scheduling_policy="priority",
                    log_path=Path(temporary_dir) / "server.log",
                )
                log_handle.close()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")
        self.assertEqual(command[command.index("--scheduling-policy") + 1], "priority")


if __name__ == "__main__":
    unittest.main()
