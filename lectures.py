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
from examples import mean_and_sigma, skewness_kurtosys
from examples import clt

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
        "title": "Probability: distributions",
        "examples": [
            {"id": "mean_and_sigma", "title": "Mean and Sigma in 1D",
             "render": mean_and_sigma.render},
            {"id": "skewness_kurtosys", "title": "Skewness and Kurtosis in 1D",
             "render": skewness_kurtosys.render},
            {"id": "clt", "title": "Central Limit Theorem",
             "render": clt.render},
        ],
    },
]
