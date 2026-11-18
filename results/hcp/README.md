# Archived HCP aggregate results

These are the saved numerical results underlying the `main_1110` and `supplement_1110` revision, not results of new fits made while preparing this repository.

- `cohort_comparison.csv`: the nine cohort/method validation AUC values from the completed M=4 comparisons.
- `candidate_dimensions.csv`: original candidate validation scores and retained draw counts. F31_35/M2 has 1,225 retained draws and M22_25/M2 has 1,175; their candidate grids were incomplete. The new runner requests 1,500 retained draws for all candidates.
- `transfer_network.csv`: fixed-source target-participant test predictions and source-only baselines, with participant-bootstrap log-score gains.
- `transfer_behavior.csv`: all seven source-specific behavioral pairings, their source/target correlations and multiplicity adjustments. Three-person target groups are descriptive only.
- `behavior_screen.csv`: all 3,516 exploratory source-cohort comparisons, including untestable comparisons in the adjustment family.
- `regional_intensities.csv`: mean absolute incident template strength and normalized prominence for all 68 ROIs in all twelve source templates.
- `behavior_groups.csv`: the fifteen aggregate group summaries in the five displayed template–behavior panels.

No subject identifiers, individual networks, memberships, behavioral values or posterior draws are included. Predictions use validation entries for the original three-cohort method comparison and reserved test entries for fixed-template transfer.

The figure CSVs were exported from the verified saved figure tables; the other CSVs come from the completed cohort comparison, exploratory screen, and transfer analyses. `hcp/DATA.md` describes their methods and how to run the analysis on user-supplied inputs.
