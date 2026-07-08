"""
generate_data.py

Builds a synthetic dataset of social media posts for the SafeX engagement
forecasting project. Real client post history wasn't available, so this
encodes well-documented real-world patterns (video/carousel beating static
images, evening posting windows beating early morning, sweet spots for
hashtag count and caption length) plus Gaussian noise, so the data behaves
the way real engagement data does.

Usage:
    python3 generate_data.py                # 2000 posts, seed 42
    python3 generate_data.py -n 5000 -s 7    # custom size / seed
"""

import argparse

import numpy as np
import pandas as pd

POST_TYPES = ["image", "video", "carousel", "text"]
PLATFORMS = ["instagram", "facebook", "linkedin", "twitter"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic social media post data.")
    parser.add_argument("-n", "--n-posts", type=int, default=2000, help="number of posts to generate")
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("-o", "--out", default="data/social_media_posts.csv", help="output CSV path")
    return parser.parse_args()


def generate(n_posts=2000, seed=42):
    rng = np.random.default_rng(seed)

    post_type = rng.choice(POST_TYPES, size=n_posts, p=[0.35, 0.30, 0.20, 0.15])
    platform = rng.choice(PLATFORMS, size=n_posts, p=[0.4, 0.3, 0.2, 0.1])
    day_of_week = rng.choice(DAYS, size=n_posts)
    hour_of_day = rng.integers(0, 24, size=n_posts)
    hashtag_count = rng.integers(0, 20, size=n_posts)
    caption_length = rng.integers(5, 400, size=n_posts)
    follower_count = rng.integers(500, 200_000, size=n_posts)
    is_weekend = np.isin(day_of_week, ["Sat", "Sun"]).astype(int)

    # engagement rate = base level + a handful of additive effects + noise
    base = rng.normal(loc=2.0, scale=0.4, size=n_posts)

    type_effect = np.select(
        [post_type == "video", post_type == "carousel", post_type == "image", post_type == "text"],
        [1.6, 1.1, 0.4, -0.3],
    )

    # evening (18-22) best, early morning (0-6) worst
    hour_effect = np.where(
        (hour_of_day >= 18) & (hour_of_day <= 22), 0.9,
        np.where((hour_of_day >= 9) & (hour_of_day <= 17), 0.3,
                 np.where((hour_of_day >= 0) & (hour_of_day <= 6), -0.8, 0.0))
    )

    weekend_effect = is_weekend * 0.35

    # sweet spots rather than "more is better" -- quadratic penalty either side
    hashtag_effect = -0.01 * (hashtag_count - 7) ** 2 + 0.6
    caption_effect = -0.00002 * (caption_length - 110) ** 2 + 0.5

    # larger accounts see a slightly lower engagement rate, not higher
    follower_effect = -0.0000015 * follower_count

    platform_effect = np.select(
        [platform == "instagram", platform == "facebook",
         platform == "linkedin", platform == "twitter"],
        [0.5, 0.1, 0.2, -0.1],
    )

    noise = rng.normal(0, 0.5, size=n_posts)

    engagement_rate = (
        base + type_effect + hour_effect + weekend_effect
        + hashtag_effect + caption_effect + follower_effect
        + platform_effect + noise
    )
    engagement_rate = np.clip(engagement_rate, 0.05, None)

    likes = (engagement_rate / 100 * follower_count * rng.uniform(0.7, 0.85, n_posts)).round().astype(int)
    comments = (engagement_rate / 100 * follower_count * rng.uniform(0.05, 0.15, n_posts)).round().astype(int)
    shares = (engagement_rate / 100 * follower_count * rng.uniform(0.05, 0.15, n_posts)).round().astype(int)

    return pd.DataFrame({
        "post_id": [f"P{i:05d}" for i in range(n_posts)],
        "platform": platform,
        "post_type": post_type,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "hashtag_count": hashtag_count,
        "caption_length": caption_length,
        "follower_count": follower_count,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "engagement_rate": engagement_rate.round(3),
    })


if __name__ == "__main__":
    args = parse_args()
    df = generate(n_posts=args.n_posts, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows -> {args.out}")
    print(df.head())
    print("\nEngagement rate summary:")
    print(df["engagement_rate"].describe())
