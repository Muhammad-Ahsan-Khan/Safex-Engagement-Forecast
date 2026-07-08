# Case study: forecasting social media engagement for SafeX Solutions

**Author:** Muhammad Ahsan Khan
**Project type:** AI/ML prototype — internship deliverable
**Stack:** Python, pandas, scikit-learn, Streamlit

## 1. Problem statement

SafeX Solutions offers social media marketing alongside web development,
cybersecurity, and SEO. One question comes up constantly for any brand's
marketing team: *before a post goes live, roughly how well is it going to
perform?* Being able to estimate that ahead of time helps decide which
posts to prioritize, when to schedule them, and what format is worth the
production effort — video versus a static image, for instance.

This project builds a small machine learning model that predicts a post's
**engagement rate** — likes plus comments plus shares, as a percentage of
the account's follower count — using only attributes known before
publishing: platform, post type, day and hour posted, hashtag count,
caption length, and follower count.

## 2. Data

SafeX didn't have a client's historical post export ready to hand over for
this prototype, so I generated a synthetic dataset of 2,000 posts instead
(`generate_data.py`). Rather than pure randomness, it's built around
patterns that show up consistently in real social media research:

- Video and carousel posts outperform static images and plain text
- Evening posts (6–10pm) beat early-morning posts (12am–6am)
- Weekend posts get a modest lift
- Both hashtag count and caption length have a "sweet spot" rather than a
  linear relationship — too few or too many of either hurts slightly
- Larger accounts tend to see a slightly *lower* engagement *rate*, which
  matches what's widely reported on real platforms

Gaussian noise is layered on top of all of that, so the dataset behaves the
way real engagement data does — a genuine signal that's still buried in
real-world unpredictability. I'm disclosing the substitution here on
purpose rather than glossing over it: the modeling approach, evaluation
methodology, and dashboard are exactly what I'd use against real client
data if it were available.

## 3. Approach

1. **EDA** — plotted engagement rate against post type and hour of day to
   confirm the patterns above were actually present and learnable, and
   added a correlation heatmap across the numeric features.
2. **Feature engineering** — bucketed posting hour into early morning /
   morning / afternoon / evening / late night, flagged weekends, and
   one-hot encoded platform, post type, day of week, and hour bucket. That
   turned the original 8 raw attributes into 21 model-ready features.
3. **Modeling** — trained three models to compare: a Linear Regression
   baseline, a Random Forest tuned over `n_estimators` and `max_depth` with
   3-fold grid search, and a Gradient Boosting Regressor.
4. **Evaluation** — compared all three on a held-out 20% test set using
   RMSE, MAE, and R², and cross-validated the Random Forest across 5 folds
   as a sanity check on the split.
5. **Delivery** — packaged the winning model into a Streamlit dashboard
   with live predictions, a model comparison view, and a filterable look at
   the training data, so it's something a marketer could actually click
   through rather than a static notebook.

## 4. Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.734 | 0.593 | 0.675 |
| Random Forest (tuned) | 0.698 | 0.568 | 0.706 |
| **Gradient Boosting** | **0.638** | **0.511** | **0.755** |

Gradient Boosting beat both the linear baseline and the tuned Random Forest
on every metric and was selected as the final model. An R² of 0.75 means
it explains about three-quarters of the variance in engagement rate from
pre-publish attributes alone — a strong result given that real engagement
always keeps an irreducible unpredictable component (content quality,
current events, algorithm changes) that no pre-publish feature set can
fully capture. As a sanity check, 5-fold cross-validation on the Random
Forest landed at R² = 0.67 ± 0.03, in the same ballpark as the single
held-out split.

**Top predictive features:**

1. Caption length (19.6% importance)
2. Hour of day (15.4%)
3. Hashtag count (14.0%)
4. Post type: text (13.2%)
5. Post type: video (13.0%)
6. Hour bucket: evening (7.3%)

## 5. Insights

- **Caption length matters more than post format.** It's the single biggest
  predictor by a comfortable margin — a strong signal that copywriting
  deserves as much attention as format choice in content planning.
- **Timing shows up twice** — both raw hour of day and the evening bucket
  rank highly, which lines up with the intuition that posting windows are
  worth planning around, not an afterthought.
- **Video is a consistent lift; text-only is a consistent drag.** Both
  show up strongly, matching the pattern built into the data and consistent
  with what's widely reported for real platforms.
- **Hashtags and captions both have a sweet spot, not a "more is better"
  relationship.** Worth testing directly against a specific client's real
  data, since the exact optimal range likely shifts by platform and
  audience.

## 6. Limitations

- Trained on synthetic, not real, engagement data — the specific numbers
  above (e.g. "19.6% importance") describe this dataset and generating
  function, not universal laws of social media.
- No content-level signal is modeled — no actual image quality, caption
  wording, or video content, only metadata about the post.
- Correlation, not causation: the model shows what's *associated* with
  higher engagement, not proof that changing one variable *causes* a change
  in another.
- Any of these patterns would need A/B testing on real posts before being
  acted on for a real client.

## 7. Next steps

- Swap the synthetic dataset for a real client's historical post export
  (most platforms support this through their analytics/export tools) and
  retrain.
- Add content-level features through NLP — caption sentiment, presence of
  a call-to-action, emoji count.
- Track prediction accuracy against real outcomes as new posts go live, to
  catch model drift as platform algorithms change.
- Extend the dashboard so a marketer can paste in a draft caption and get
  hashtag and timing recommendations back, not just a single number.

---
All code, the trained model, and the interactive dashboard are in this
repository. See `README.md` to run it locally.
