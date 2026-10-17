# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""A support-ticket triage heuristic: flag a ticket as urgent when its weighted score passes a
threshold. OmniEvolve tunes the constants between the EVOLVE-BLOCK markers; everything else
(the feature names and the function signature) stays fixed.
"""

# EVOLVE-BLOCK-START
WEIGHT_DAYS_OPEN = 0.1  # per day the ticket has been open
WEIGHT_TIER = 0.5  # 0 = free, 1 = pro, 2 = enterprise
WEIGHT_SENTIMENT = -0.5  # sentiment runs from -1 (angry) to 1 (happy)
WEIGHT_OUTAGE = 1.0  # 1 when the customer reports an outage
WEIGHT_REPLIES = 0.5  # replies so far in the thread
THRESHOLD = 4.0
# EVOLVE-BLOCK-END


def urgency(ticket: dict) -> float:
    return (
        WEIGHT_DAYS_OPEN * ticket["days_open"]
        + WEIGHT_TIER * ticket["tier"]
        + WEIGHT_SENTIMENT * ticket["sentiment"]
        + WEIGHT_OUTAGE * ticket["outage"]
        + WEIGHT_REPLIES * ticket["replies"]
    )


def is_urgent(ticket: dict) -> bool:
    return urgency(ticket) > THRESHOLD
