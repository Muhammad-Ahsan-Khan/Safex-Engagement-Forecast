# SafeX Engagement Forecast

Predicting how well a social media post will perform *before it goes live* —
built as an AI/ML internship deliverable for SafeX Solutions, a digital
agency offering social media marketing, web development, cybersecurity, and
SEO services.

Given a handful of attributes known ahead of publishing (platform, post
type, posting time, hashtag count, caption length, follower count), the
model estimates **engagement rate**: `(likes + comments + shares) / followers`,
expressed as a percentage. A Streamlit dashboard wraps the model so
predictions can be explored interactively instead of read off a notebook.

## Results at a glance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.734 | 0.593 | 0.675 |
| Random Forest (tuned) | 0.698 | 0.568 | 0.706 |
| **Gradient Boosting** | **0.638** | **0.511** | **0.755** |

Gradient Boosting came out on top and explains roughly 75% of the variance
in engagement rate using only pre-publish attributes. Full breakdown,
feature importances, and discussion in [`case_study.md`](case_study.md).

## Project structure

```
safex-engagement-forecast/
├── app.py                    # Streamlit dashboard
├── generate_data.py          # Synthetic dataset generator
├── train_model.py            # Feature engineering, training, evaluation
├── case_study.md             # Full write-up: problem, approach, results, insights
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # + pytest, for running the test suite
├── tests/
│   └── test_pipeline.py      # Sanity checks on data generation & features
├── data/
│   ├── social_media_posts.csv
│   └── test_predictions.csv
├── models/
│   ├── model.pkl
│   └── feature_columns.json
├── images/                   # Charts produced during training
└── .github/workflows/tests.yml
```

## Getting started

```bash
# 1. Clone and set up an environment
git clone https://github.com/<your-username>/safex-engagement-forecast.git
cd safex-engagement-forecast
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. Generate the dataset
python3 generate_data.py

# 3. Train the models (writes charts, models/model.pkl, results.json)
python3 train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

## The dashboard

Five tabs:

- **Predict** — pick post attributes on the left, get a live engagement
  forecast, an estimated like/comment/share split, a percentile against the
  training set, and a quick read on what's helping or hurting the number.
- **Compare scenarios** — configure two posts side by side and see which one
  is forecast to perform better and by how much.
- **Batch predict** — upload a CSV of draft posts and get engagement
  forecasts for all of them at once, with a downloadable results file.
- **Model performance** — side-by-side metrics for all three models plus the
  evaluation charts from training.
- **Explore the data** — the EDA charts, a best-time-to-post heatmap by
  platform and hour, and a filterable table of the underlying posts.

## A note on the data

Real historical post data from a SafeX client wasn't available while this
was built, so `generate_data.py` produces a synthetic dataset instead. It
isn't random — it's built around patterns that are well documented on real
platforms (video and carousel posts outperforming static images, evening
posting windows beating early morning, hashtag count and caption length
each having a "sweet spot" rather than a linear relationship), with noise
layered on top so the data behaves the way real engagement data does. This
is disclosed in full in `case_study.md` rather than hidden, and the
modeling/evaluation approach is exactly what would be used against a real
client dataset.

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · seaborn

## License

MIT — see [`LICENSE`](LICENSE).

## Author

Muhammad Ahsan Khan
