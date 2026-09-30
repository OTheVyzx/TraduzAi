PRAGMA foreign_keys = ON;

CREATE TABLE repositories (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  signer_fingerprint TEXT NOT NULL,
  trusted_at TEXT NOT NULL,
  last_refreshed_at TEXT
);

CREATE TABLE extension_versions (
  package_name TEXT NOT NULL,
  version_code INTEGER NOT NULL,
  version_name TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  signer_fingerprint TEXT NOT NULL,
  repository_id TEXT NOT NULL REFERENCES repositories(id),
  state TEXT NOT NULL CHECK (state IN ('quarantined','active','rollback','rejected')),
  PRIMARY KEY (package_name, version_code)
);

CREATE TABLE sources (
  id TEXT PRIMARY KEY,
  package_name TEXT NOT NULL,
  source_id TEXT NOT NULL,
  name TEXT NOT NULL,
  language TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  UNIQUE (package_name, source_id)
);

CREATE TABLE reader_manga (
  id TEXT PRIMARY KEY,
  source_record_id TEXT NOT NULL REFERENCES sources(id),
  manga_url TEXT NOT NULL,
  memo BLOB,
  title TEXT NOT NULL,
  canonical_url TEXT,
  metadata_json TEXT NOT NULL,
  in_library INTEGER NOT NULL DEFAULT 0,
  auto_update INTEGER NOT NULL DEFAULT 0,
  auto_download INTEGER NOT NULL DEFAULT 0,
  UNIQUE (source_record_id, manga_url)
);

CREATE TABLE reader_chapters (
  id TEXT PRIMARY KEY,
  manga_id TEXT NOT NULL REFERENCES reader_manga(id) ON DELETE CASCADE,
  chapter_url TEXT NOT NULL,
  memo BLOB,
  metadata_json TEXT NOT NULL,
  read INTEGER NOT NULL DEFAULT 0,
  last_page INTEGER NOT NULL DEFAULT 0,
  bookmarked INTEGER NOT NULL DEFAULT 0,
  UNIQUE (manga_id, chapter_url)
);

CREATE TABLE categories (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, position INTEGER NOT NULL);
CREATE TABLE manga_categories (
  manga_id TEXT NOT NULL REFERENCES reader_manga(id) ON DELETE CASCADE,
  category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
  PRIMARY KEY (manga_id, category_id)
);

CREATE TABLE downloads (
  id TEXT PRIMARY KEY,
  chapter_id TEXT NOT NULL REFERENCES reader_chapters(id) ON DELETE CASCADE,
  state TEXT NOT NULL,
  manifest_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  state TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  due_at TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE source_preferences (source_record_id TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE, value_json TEXT NOT NULL);
CREATE TABLE extension_secrets (package_name TEXT PRIMARY KEY, encrypted_value BLOB NOT NULL);
