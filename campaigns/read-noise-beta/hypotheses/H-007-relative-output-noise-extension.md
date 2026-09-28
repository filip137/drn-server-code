---
id: "H-007"
title: "Ours has an average advantage over legacy across extended displacement and noise"
---
# H007 — Extended displacement and endpoint noise

Filip added output displacement2.5/3 and additive noise1e-3/2e-3 after partial
exp007 results on September24. This extends H006 without rewriting it.
Before extension production, Filip restricted Conv2/3 to targets strictly>1.6
and then added target6.0 for all three architectures.
The18 selected reused cells were already partially observed;117 new trainings are
prospective. All original exp007 results remain separately reported.

Under the exp008 held-fixed MNIST EqProp contract, compare ours−legacy epoch10
validation accuracy at seven initial D/F targets for Conv1 and four (2,2.5,3,6)
for Conv2/3, at three absolute endpoint-noise levels. Report45 paired differences.
For each architecture/noise, positive equal-weight mean across all declared
complete finite pairs (seven for Conv1, four for Conv2/3) supports the scoped average
advantage; negative mean contradicts; exact tie or missing pair is inconclusive.
Report sign consistency, baseline and losses; no independent best-target selection.
All-depth/all-noise support requires every scoped mean positive and complete.
Single seed does not establish significance or seed-general superiority.

Do not interpret initial D/F matching as equal absolute signal/noise, or absolute
Gaussian endpoint noise as relative Gaussian noise. Report source/host deviations
and retain cancelled/failed cases. No post-hoc threshold or missing-pair averaging.
