# TREC-IS 2020-A Dataset (AI-03 Development Data)

This directory contains the official dataset files for **AI-03: Crisis Report Fusion and Priority Ranking**.

## 1. Provenance and Official Source
- **Benchmark Track**: Text REtrieval Conference (TREC) Incident Streams Track (TREC-IS)
- **Official Organizers**: University of Glasgow & National Institute of Standards and Technology (NIST)
- **Track Portal**: [http://trecis.org/](http://trecis.org/) / [https://trec.nist.gov/data/incident.html](https://trec.nist.gov/data/incident.html)
- **Target Edition**: TREC Incident Streams 2020-A (`trecis2020-A`), Events 35–49
- **Official Client**: `TREC-IS-DatasetClient-4.1.jar`
  - URL: `https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/2021A/TREC-IS-DatasetClient-4.1.jar`
- **Official Annotations**: `TRECIS-2018-2020B.json.gz` (Events 1–75, covers 35–49)
  - URL: `https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/TRECIS-2018-2020B.json.gz`

## 2. Directory Layout
```text
data/
├── raw/                                  # Ignored from git
│   ├── TREC-IS-DatasetClient-4.1.jar     # Official downloader client
│   ├── info.json                         # Client configuration (request: trecis2020-A)
│   ├── TRECIS-2018-2020B.json.gz         # Official ground-truth labels
│   └── trecis2020-A/                     # Per-event raw tweet streams (.json.gz)
├── processed/                            # Ignored from git
│   ├── trecis2020_a_clean.jsonl          # Filtered original messages with labels
│   ├── trecis2020_a_variants.jsonl       # Generated deterministic variants
│   └── trecis2020_a_dev_full.jsonl       # Combined development set (clean + variants)
└── README.md                             # This documentation
```

## 3. Preparation Pipeline
The data preparation script `scripts/prepare_p1_dataset.py`:
1. Acquires the official client and ground-truth labels directly from the official Glasgow / TREC-IS servers.
2. Runs the official client with `request: trecis2020-A` to retrieve raw tweets for events 35–49.
3. Filters for records meeting both required criteria:
   - A valid Task-2 reduced information-category label.
   - A valid priority or criticality label (`Critical`, `High`, `Medium`, `Low`).
4. Generates deterministic variants for message ID $m$ using the official specification:
   - $h = \text{SHA256}("20260911:" + m)$
   - $h_0 \pmod{10} \in \{0, 1, 2, 3, 4, 5\}$
   - Transformation chosen via $h_1 \pmod 4$.

## 4. Development Record Schema
```json
{
  "item_id": "1234567890",
  "text": "Flash flood waters rising fast...",
  "event_id": "TRECIS-CTIT-H-Test-035",
  "category": "SearchAndRescue",
  "priority": "High",
  "priority_score": 4.0,
  "is_variant": false,
  "source_item_id": "1234567890",
  "transform_type": null
}
```

## 5. Development Ground Truth Definition for Clustering Evaluation
For offline Pairwise Clustering F1 evaluation on development data:
- **Original ↔ Original**: Two original messages refer to the same underlying incident if and only if they belong to the same incident event: `event_id_A == event_id_B`.
- **Original ↔ Variant**: A variant $v$ and an original message $m$ represent the exact same underlying incident and crisis information if `v.source_item_id == m.item_id`.
- **Variant ↔ Variant**: Two variants refer to the same underlying information if either:
  1. They derive from the same source message (`v1.source_item_id == v2.source_item_id`), OR
  2. They derive from distinct source messages belonging to the same incident (`v1.event_id == v2.event_id`).

*Note: Raw and processed datasets are ignored by version control to comply with reproducibility and data redistribution policies.*
