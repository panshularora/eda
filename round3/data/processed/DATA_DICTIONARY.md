# Round 3 dataset - Reaction to a Major Delivery or Service Delay

**76,123 reactions** collected from public sources, 2026-08-06 to 2026-09-20 (44.99 days).

| column | type | provenance | complete | distinct | meaning |
|---|---|---|---|---|---|
| `record_id` | object | derived | 100% | 76,123 | SHA-256 of source + native id + text prefix; stable across runs |
| `source` | object | collected | 100% | 5 | platform the reaction came from |
| `source_id` | object | collected | 100% | 99 | app id, subreddit, hashtag or query that produced the row |
| `brand` | object | collected/derived | 100% | 462 | operator the reaction is about; from the app roster, or matched in text for social sources |
| `delay_domain` | object | collected/derived | 100% | 8 | sector of the delay: food_delivery, quick_commerce, parcel_courier, ecommerce, ride_hailing, airline, telecom_isp |
| `store_country` | object | collected | 100% | 8 | storefront the review was read from (Play only) |
| `is_thin_text` | bool | derived | 100% | 2 | text shorter than 15 characters; still valid for volume and rating, too thin to mine |
| `created_utc` | datetime64[ns, UTC] | collected | 100% | 74,507 | when the reaction was published, in UTC |
| `date` | object | derived | 100% | 46 | UTC calendar date of created_utc |
| `hour_utc` | object | derived | 100% | 1,081 | UTC hour bucket of created_utc |
| `collected_utc` | object | derived | 100% | 141 | when our collector retrieved the row |
| `title` | object | collected | 100% | 932 | headline or post title; empty for Play reviews |
| `text` | object | collected | 100% | 76,094 | body text as published, whitespace-normalised, URLs stripped |
| `full_text` | object | derived | 100% | 76,123 | title + text, the field all NLP runs on |
| `text_length` | int64 | derived | 100% | 925 | characters in full_text |
| `word_count` | int64 | derived | 100% | 322 | whitespace tokens in full_text |
| `author_pseudonym` | object | derived | 100% | 70,184 | salted SHA-256 of the handle; the raw handle is never stored |
| `rating` | float64 | collected | 97% | 5 | 1-5 star rating chosen by the reviewer (Play only) - an independent sentiment label |
| `engagement` | int32 | collected | 100% | 164 | endorsement count; meaning varies by source, see engagement_kind |
| `engagement_kind` | object | derived | 100% | 4 | what engagement counts on this source: thumbs_up, fav_boost_reply, points_comments or none |
| `app_version` | object | collected | 100% | 2,269 | app version the reviewer was running (Play only) |
| `company_replied` | bool | collected | 100% | 2 | whether the operator publicly replied |
| `company_reply_utc` | object | collected | 100% | 24,358 | when the operator replied |
| `url` | object | collected | 100% | 2,148 | public link to the reaction or its source |
| `is_delay_related` | bool | derived | 100% | 2 | text matches the delay-relevance pattern |
| `delay_type` | object | derived | 100% | 11 | first matching delay pattern; see delay_type_evidence |
| `delay_type_evidence` | object | derived | 100% | 741 | the literal span that triggered delay_type - makes the label auditable |
| `reaction_type` | object | derived | 100% | 9 | first matching reaction pattern; see reaction_type_evidence |
| `reaction_type_evidence` | object | derived | 100% | 255 | the literal span that triggered reaction_type |
| `stated_delay_hours` | float64 | derived | 5% | 137 | largest duration mentioned in the text, in hours; crude severity proxy |
| `r2_sentiment` | object | model | 100% | 3 | Round 2 sentiment model prediction: Negative / Neutral / Positive |
| `r2_sentiment_confidence` | float64 | model | 100% | 6,380 | calibrated probability of the predicted sentiment class |
| `r2_topic` | object | model | 100% | 4 | Round 2 topic model prediction |
| `r2_topic_confidence` | float64 | model | 100% | 4,340 | calibrated probability of the predicted topic class |
| `sentiment_score` | float64 | derived | 100% | 3 | sentiment mapped to -1 / 0 / +1 for time-series arithmetic |
| `sentiment_score_weighted` | float64 | derived | 100% | 11,578 | sentiment_score multiplied by model confidence |

## Companion tables

| file | rows | contents |
|---|---|---|
| `round3_incidents.csv` | 244 | documented delay events with start times |
| `round3_attention.csv` | 1,440 | daily Wikipedia pageviews per brand |
