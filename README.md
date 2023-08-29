# Dynamic Noise Estimation

Choice data in decision-making is inherently noisy, and isoloting noise from signal typically improves the quality of computational modeling. As of right now, a majority of models measure noise as static/constant, through a single parameter. However, this assumption does not always hold, as agents can lapse into inattentive phases for a series of trials in the middle of otherwise low-noise performance. Here, we propose a method to dynamically infer noise in choice behavior, under the assumption that agents can transition between two discrete latent states. We demonstrate that our method improves model fit and it can easily be incorporated into existing fitting procedures, such as MLE and hierarchical Bayesian modeling.

