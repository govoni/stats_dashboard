"""
Registry of lectures and their examples.

This is the only file you edit to add content:
  - to add an example, write examples/your_example.py with a render()
    function (copy examples/_template.py to start), then add one entry
    to the relevant lecture's "examples" list below.
  - to add a lecture, add a new dict to LECTURES.

Each example entry is:
    {"id": "<unique-str>", "title": "<button label>", "render": <callable>}

"render" must be a zero-argument callable. For real examples this is
just the module's render function. For lectures you haven't built yet,
use `placeholder_for(title)` as a drop-in stand-in.
"""

import functools

from examples import dice_frequentist, uniform_sampling, uniform_sampling_2D, uniform_cumulative, non_uniform_sampling
from examples import mean_and_sigma, skewness_kurtosys, clt, mean_uncertainty, mean_uncertainty_2
from examples import histogram_sampler
from examples import change_of_variables
from examples import bivariate_pdf, gauss_2D
from examples import resonance_over_fluctuations

def placeholder_for(title):
    """Bind the generic placeholder module to a specific example title."""
    return functools.partial(placeholder.render, title=title)


LECTURES = [
    {
        "title": "Probability: foundations",
        "examples": [
            {"id": "dice_frequentist", "title": "Rolling a die",
             "render": dice_frequentist.render},
            {"id": "uniform_sampling", "title": "Uniform sampling in 1D",
             "render": uniform_sampling.render},
            {"id": "uniform_sampling_2D", "title": "Uniform sampling in 2D",
             "render": uniform_sampling_2D.render},
            {"id": "uniform_cumulative", "title": "Cumulative uniform sampling in 1D",
             "render": uniform_cumulative.render},
            {"id": "non_uniform_sampling", "title": "Non-uniform sampling in 1D",
             "render": non_uniform_sampling.render},
        ],
    },
    {
        "title": "Probability: 1D continuous distributions",
        "examples": [
            {"id": "mean_and_sigma", "title": "Mean and Sigma in 1D",
             "render": mean_and_sigma.render},
            {"id": "skewness_kurtosys", "title": "Skewness and Kurtosis in 1D",
             "render": skewness_kurtosys.render},
            {"id": "clt", "title": "Central Limit Theorem",
             "render": clt.render},
            {"id": "mean_uncertainty", "title": "Single Measure and Mean Uncertainty",
             "render": mean_uncertainty.render},
            {"id": "mean_uncertainty_2", "title": "Single Measure and Mean Uncertainty V2",
             "render": mean_uncertainty_2.render},
            {"id": "change_of_variables", "title": "1D change of variables",
             "render": change_of_variables.render},
        ],
    },
    {
        "title": "Probability: 2D continuous distributions",
        "examples": [
            {"id": "bivariate_pdf", "title": "2D distributions",
             "render": bivariate_pdf.render},
            {"id": "gauss_2D", "title": "2D Gaussian",
             "render": gauss_2D.render},
        ],
    },
    {
        "title": "Probability: discrete distributions",
        "examples": [
            {"id": "histogram_sampler", "title": "Histogram bin distribution",
             "render": histogram_sampler.render},
            {"id": "resonance_over_fluctuations", "title": "Resonance hunting",
             "render": resonance_over_fluctuations.render},
        ],
    },
]
