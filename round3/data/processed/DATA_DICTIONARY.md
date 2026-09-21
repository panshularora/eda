# Round 3 dataset - Reaction to a Major Delivery or Service Delay

**92,689 reactions** collected from public sources, 2026-08-06 to 2026-09-20 (45.0 days).

| column | type | provenance | complete | distinct | meaning |
|---|---|---|---|---|---|
| `record_id` | object | derived | 100% | 92,689 | SHA-256 of source + native id + text prefix; stable across runs |
| `source` | object | collected | 100% | 7 | platform the reaction came from |
| `source_id` | object | collected | 100% | 180 | app id, subreddit, hashtag or query that produced the row |
| `brand` | object | collected/derived | 100% | 45 | operator the reaction is about; from the app roster, or matched in text for social sources |
| `delay_domain` | object | collected/derived | 100% | 8 | sector of the delay: food_delivery, quick_commerce, parcel_courier, ecommerce, ride_hailing, airline, telecom_isp |
| `store_country` | object | collected | 100% | 8 | storefront the review was read from (Play only) |
| `is_thin_text` | bool | derived | 100% | 2 | text shorter than 15 characters; still valid for volume and rating, too thin to mine |
| `created_utc` | datetime64[ns, UTC] | collected | 100% | 89,734 | when the reaction was published, in UTC |
| `date` | object | derived | 100% | 46 | UTC calendar date of created_utc |
| `hour_utc` | object | derived | 100% | 1,081 | UTC hour bucket of created_utc |
| `collected_utc` | object | derived | 100% | 290 | when our collector retrieved the row |
| `title` | object | collected | 100% | 3,286 | headline or post title; empty for Play reviews |
| `text` | object | collected | 100% | 66,344 | body text as published, whitespace-normalised, URLs stripped |
| `full_text` | object | derived | 100% | 66,069 | title + text, the field all NLP runs on |
| `text_length` | int64 | derived | 100% | 1,196 | characters in full_text |
| `word_count` | int64 | derived | 100% | 507 | whitespace tokens in full_text |
| `author_pseudonym` | object | derived | 100% | 82,929 | salted SHA-256 of the handle; the raw handle is never stored |
| `publisher` | object | collected | 5% | 1,183 | outlet that published the article (news rows only); never a delay brand |
| `rating` | float64 | collected | 95% | 5 | 1-5 star rating chosen by the reviewer (Play only) - an independent sentiment label |
| `engagement` | int32 | collected | 100% | 162 | endorsement count; meaning varies by source, see engagement_kind |
| `engagement_kind` | object | derived | 100% | 5 | what engagement counts on this source: thumbs_up, fav_boost_reply, points_comments or none |
| `app_version` | object | collected | 100% | 2,331 | app version the reviewer was running (Play only) |
| `company_replied` | bool | collected | 100% | 2 | whether the operator publicly replied |
| `company_reply_utc` | object | collected | 100% | 33,578 | when the operator replied |
| `url` | object | collected | 100% | 4,724 | public link to the reaction or its source |
| `is_delay_related` | bool | derived | 100% | 2 | text matches the delay-relevance pattern |
| `delay_type` | object | derived | 100% | 11 | first matching delay pattern; see delay_type_evidence |
| `delay_type_evidence` | object | derived | 100% | 769 | the literal span that triggered delay_type - makes the label auditable |
| `reaction_type` | object | derived | 100% | 9 | first matching reaction pattern; see reaction_type_evidence |
| `reaction_type_evidence` | object | derived | 100% | 254 | the literal span that triggered reaction_type |
| `stated_delay_hours` | float64 | derived | 4% | 154 | largest duration mentioned in the text, in hours; crude severity proxy |
| `r2_sentiment` | object | model | 100% | 3 | Round 2 sentiment model prediction: Negative / Neutral / Positive |
| `r2_sentiment_confidence` | float64 | model | 100% | 6,347 | calibrated probability of the predicted sentiment class |
| `r2_p_negative` | float64 | model | 100% | 9,372 | Round 2 model P(Negative); kept because the max alone cannot support recalibration |
| `r2_p_neutral` | float64 | model | 100% | 8,017 | Round 2 model P(Neutral) |
| `r2_p_positive` | float64 | model | 100% | 9,607 | Round 2 model P(Positive) |
| `r2_topic` | object | model | 100% | 4 | Round 2 topic model prediction. NOT INTERPRETED: Round 2 established this label is a substring switch, and reports/analysis.json measures how closely the model still reproduces it on this corpus. Shipped because the rulebook requires the Round 2 model to be applied. |
| `r2_topic_confidence` | float64 | model | 100% | 4,158 | calibrated probability of the predicted topic class |
| `flag_churn_threat` | bool | derived | 100% | 2 | mentions uninstalling, switching or cancelling a subscription; independent of reaction_type |
| `flag_refund_demand` | bool | derived | 100% | 2 | asks for money back or compensation; independent of reaction_type |
| `flag_escalation` | bool | derived | 100% | 2 | threatens a complaint, consumer forum or legal action |
| `flag_anger` | bool | derived | 100% | 2 | uses abusive or outraged vocabulary |
| `flag_recovery` | bool | derived | 100% | 2 | mentions the issue being resolved, refunded or apologised for |
| `flag_repeat_incident` | bool | derived | 100% | 2 | says this has happened before |
| `flag_money_lost` | bool | derived | 100% | 2 | says money was charged, deducted or lost |
| `flag_staff_blamed` | bool | derived | 100% | 2 | names a driver, rider, courier or agent |
| `competitor_named` | object | derived | 100% | 765 | the operator the customer says they are switching to, if any |
| `sentiment_score` | float64 | derived | 100% | 3 | sentiment mapped to -1 / 0 / +1 for time-series arithmetic |
| `sentiment_score_weighted` | float64 | derived | 100% | 11,351 | sentiment_score multiplied by model confidence |

## Companion tables

| file | rows | contents |
|---|---|---|
| `round3_incidents.csv` | 194 | documented delay events with start times |
| `round3_attention.csv` | 1,440 | daily Wikipedia pageviews per brand |
