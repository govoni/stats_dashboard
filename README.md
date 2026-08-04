# Statistics & ML course dashboard

## Run

```
pip install streamlit numpy scipy matplotlib
streamlit run app.py
```

## Structure

```
app.py                         navigation shell only (lecture -> example -> full-screen demo)
lectures.py                    registry: 12 lectures, each with a list of examples
examples/
    _template.py                copy this to start a new example
    placeholder.py               generic "not implemented yet" screen
```

## Adding a new example

1. Copy `examples/_template.py` to `examples/your_example.py` and implement `render()`.
   - Put controls in a row of `st.columns(...)` above the plot, not a side column
     next to it, so the full-screen view doesn't waste space.
   - Don't draw a title/back button yourself -- `app.py` already shows the
     breadcrumb and back button above whatever `render()` draws.
2. In `lectures.py`, import your module at the top and replace the relevant
   `placeholder_for(...)` entry with `"render": your_example.render`.

## Adding a new lecture

Add a new dict to the `LECTURES` list in `lectures.py`, following the existing
pattern. Use `placeholder_for("Example title")` for examples you haven't
built yet -- they'll show a "not implemented" screen instead of crashing.

## Navigation model

All navigation lives in `st.session_state` (`view`, `lecture_idx`,
`example_id`) inside `app.py`. You should not need to touch `app.py` again
once the shell works -- all future changes are new files under `examples/`
plus one-line registrations in `lectures.py`.
