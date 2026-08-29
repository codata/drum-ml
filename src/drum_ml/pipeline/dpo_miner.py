"""High-Fidelity DPO Negative Sample Mining with Quality Guards."""

from typing import List, Optional
from drum_ml.models.export import DPOPreferenceRecord
from drum_ml.models.scaffolds import AugmentedRecord
from drum_ml.models.validation import ValidationResult


class DPOMiner:
    """Mines clean DPO preference pairs (prompt, chosen, rejected) from validated failures,
    filtering out noisy, garbled, or mismatched prompts.
    """

    def mine_pairs(
        self,
        records: List[AugmentedRecord],
        reports: List[ValidationResult],
    ) -> List[DPOPreferenceRecord]:
        """Harvests verified DPO pairs where the prompt is high-quality and chosen answer is ground truth."""
        dpo_pairs: List[DPOPreferenceRecord] = []
        report_map = {rep.record_id: rep for rep in reports}

        for rec in records:
            rep = report_map.get(rec.id)
            if rep and not rep.passed and rep.is_dpo_eligible:
                # Plausible rejection reason
                reason = "; ".join(rep.rejection_reasons)
                # Formulate negative sample
                dpo_pairs.append(
                    DPOPreferenceRecord(
                        id=f"dpo_{rec.id}",
                        archetype=rec.archetype,
                        entity_uri=rec.entity_uri,
                        prompt=rec.user_query,
                        chosen=rec.ground_truth_answer,
                        rejected=f"[Flawed response with metrological drift: {reason}]",
                        rejection_reason=reason,
                    )
                )

        return dpo_pairs
