"""Live evaluation endpoints (single run, all runs, and error rows)."""

from __future__ import annotations

from typing import Optional

from lyra_ocr.dataset import registry
from lyra_ocr.eval.dataset_eval import evaluate_all_runs, evaluate_run
from lyra_ocr.eval.loader import load_artifacts_evaluation_items, load_golden_dataset
from lyra_ocr.eval.metrics import evaluate_predictions
from lyra_ocr.eval.report import build_prediction_rows
from lyra_ocr.web.base import _safe_resolve


class EvalAPIMixin:
    """Eval tab endpoints."""

    def _handle_get_eval(
        self,
        golden_dir_str: str | None,
        art_dir_str: str | None,
        dataset_str: str | None = None,
        run_id: str | None = None,
    ) -> None:
        try:
            if dataset_str:
                ds_path = registry.dataset_dir(self._workspace_root(), dataset_str)
                if not ds_path.is_dir():
                    self._send_error(f"Dataset not found: {dataset_str}", 404)
                    return
                if run_id == "all":
                    results = evaluate_all_runs(ds_path)
                    self._send_json(
                        {
                            "mode": "all",
                            "reports": [
                                {"run_id": item["run_id"], "report": item["report"].to_dict()}
                                for item in results
                            ],
                        }
                    )
                    return
                if not run_id:
                    self._send_error("Missing query parameter: run", 400)
                    return
                rep, _ = evaluate_run(ds_path, run_id)
                self._send_json({"mode": "single", "run_id": run_id, "report": rep.to_dict()})
                return

            if golden_dir_str:
                g_path = _safe_resolve(golden_dir_str, base_dir=self._workspace_root())
                if not g_path or not g_path.exists():
                    self._send_error(f"Golden dataset directory not found: {golden_dir_str}", 404)
                    return
                items = load_golden_dataset(g_path)
            elif art_dir_str:
                art_path = _safe_resolve(art_dir_str, base_dir=self._workspace_root())
                if not art_path or not art_path.exists():
                    self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
                    return
                items = load_artifacts_evaluation_items(art_path)
            else:
                self._send_error(
                    "Must specify dataset+run, golden, or dir query parameter", 400
                )
                return

            rep = evaluate_predictions(items)
            self._send_json(rep.to_dict())
        except Exception as e:
            self._send_error(f"Evaluation failed: {e}", 500)

    def _handle_get_eval_errors(
        self,
        golden_dir_str: Optional[str],
        art_dir_str: Optional[str],
        dataset_str: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        try:
            if dataset_str:
                ds_path = registry.dataset_dir(self._workspace_root(), dataset_str)
                if not ds_path.is_dir():
                    self._send_error(f"Dataset not found: {dataset_str}", 404)
                    return
                if not run_id or run_id == "all":
                    self._send_error("Missing query parameter: run", 400)
                    return
                rep, items = evaluate_run(ds_path, run_id)
            elif golden_dir_str:
                g_path = _safe_resolve(golden_dir_str, base_dir=self._workspace_root())
                if not g_path or not g_path.exists():
                    self._send_error(f"Golden dataset directory not found: {golden_dir_str}", 404)
                    return
                items = load_golden_dataset(g_path)
                rep = evaluate_predictions(items)
            elif art_dir_str:
                art_path = _safe_resolve(art_dir_str, base_dir=self._workspace_root())
                if not art_path or not art_path.exists():
                    self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
                    return
                items = load_artifacts_evaluation_items(art_path)
                rep = evaluate_predictions(items)
            else:
                self._send_error(
                    "Must specify dataset+run, golden, or dir query parameter", 400
                )
                return

            rows = build_prediction_rows(items)
            error_rows = [
                r for r in rows
                if r["error_type"] in ("leakage", "false_drop", "misclassification", "tracking_miss", "missing_track")
            ]
            self._send_json({"report": rep.to_dict(), "errors": error_rows})
        except Exception as exc:  # noqa: BLE001
            self._send_error(f"Evaluation failed: {exc}", 500)
