PRAGMA foreign_keys = ON;

CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    repo_id    TEXT NOT NULL,
    revision   TEXT NOT NULL
);

CREATE TABLE pairs (
    pair_id      TEXT PRIMARY KEY,
    dataset_id   INTEGER NOT NULL,
    official_id  TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    target_path  TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE (dataset_id, official_id)
);
