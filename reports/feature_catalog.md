# Reusable Feature Catalog

| Feature Name | Entity | Data Type | Formula / Logic | Description |
| :--- | :--- | :--- | :--- | :--- |
| `user_rating_count` | User | INT | COUNT(ratings in train) | Total interactions by user |
| `user_avg_rating` | User | FLOAT | MEAN(ratings in train) | User's rating leniency baseline |
| `user_preferred_genre` | User | STRING | MODE(genres of rated movies) | User's top preferred genre |
| `days_since_last_interaction` | User | FLOAT | (max_snapshot_ts - last_ts) / 86400 | Recency & churn indicator |
| `item_rating_count` | Item | INT | COUNT(ratings in train) | Movie popularity volume |
| `item_avg_rating` | Item | FLOAT | MEAN(ratings in train) | Perceived quality of movie |
| `item_popularity_score` | Item | FLOAT | count / max(count) | Normalized popularity index [0, 1] |
| `interaction_hour` | Interaction | INT | EXTRACT(HOUR from timestamp) | Time-of-day contextual consumption |
| `user_item_genre_match` | Interaction | INT | 1 if preferred_genre in item_genres else 0 | Thematic alignment match |
