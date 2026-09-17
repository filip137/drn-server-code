# Final readout protocol

Freeze schedules using the declared validation screen and any qualifying boundary extensions. Complete the frozen schedule checks on two additional saved arrays. Then evaluate their selected states, the paired P0 states, and previous one-epoch references on the same official 10,000-example MNIST test cohort. These test measurements do not select schedules, epochs, or checkpoints.

The DRN reproductions intentionally retain each original deployment's `data_order` seed. The legacy loader uses that seed for both the 55,000/5,000 stratified split and minibatch shuffling, so validation cohorts differ across DRN replicas and from the teacher/HWA split. Within each replica, every tested rate/schedule sees the same split and minibatch sequence. Final test replay uses the common official test set to avoid comparing different validation cohorts. Crossbar data splitting/shuffling uses the fixed original seed 42.

Readout uses the original forward/evaluation functions and held-apparent state, with hidden persistent state reported separately. Every readout verifies before/after state hashes and zero physical writes.

For a stronger one-epoch comparison, additionally identify the lowest validation KL observed within the first epoch among screened constant rates (including the DRN quarter-epoch checks and P0), freeze that rate separately for healthy/faulted states, and evaluate it on the two extra arrays if it differs from the previous figure's one-epoch reference. Reuse an existing reference only when the exact rate, P0, data stream, and writer contract agree. This rule is declared before inspecting any new-array result.
