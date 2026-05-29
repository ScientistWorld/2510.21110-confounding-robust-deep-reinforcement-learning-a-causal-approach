# Problem

Learn a policy from off-policy trajectories collected by a demonstrator whose actions are affected by variables hidden from the learner. The learner observes state, action, reward, and next observed state, but not the latent factor that jointly influences the demonstrator action and the observed outcome.

The central challenge is that standard off-policy Q-learning treats the logged action-outcome association as causal. Under unobserved confounding, this can make actions look better in the log than they are under intervention.

This reproduction uses a small tracked finite-horizon CMDP with two actions. One action is safe and has a modest stable reward. The other is risky and is profitable only in a hidden favorable context. The demonstrator sees that hidden context and tends to choose the risky action only when it is favorable, so the offline data is biased.

