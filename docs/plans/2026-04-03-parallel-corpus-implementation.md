# Parallel Corpus And Example Operation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first reusable EN/PT-BR corpus artifacts from the uploaded example chapters so later training and calibration work has a clean data foundation.

**Architecture:** A new Python corpus layer scans chapter archives, pairs them by normalized chapter number, computes lightweight metadata, and writes stable JSON artifacts under a dedicated dataset directory. Tests validate pairing and artifact summaries before runtime integration work begins.

**Tech Stack:** Python 3.12, unittest, zipfile/pathlib/json.

---

### Task 1: Add failing tests for chapter pairing

**Files:**
- Create: `T:\mangatl\pipeline\tests\test_parallel_corpus.py`

**Step 1: Write the failing test**

Test:
- PT-BR filename parser extracts chapter number and source label
- EN filename parser extracts chapter number
- pairing logic matches chapter 82 PT-BR with chapter 82 EN

**Step 2: Run test to verify it fails**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest tests.test_parallel_corpus -v`
Expected: FAIL because the corpus module does not exist yet.

**Step 3: Write minimal implementation**

Create the corpus parser and pairing helpers.

**Step 4: Run test to verify it passes**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest tests.test_parallel_corpus -v`
Expected: PASS

### Task 2: Add artifact builders

**Files:**
- Create: `T:\mangatl\pipeline\corpus\__init__.py`
- Create: `T:\mangatl\pipeline\corpus\parallel_dataset.py`

**Step 1: Write the failing test**

Test:
- manifest includes paired chapters
- quality profile includes total chapters and source distribution
- alignment profile includes page count delta

**Step 2: Run test to verify it fails**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest tests.test_parallel_corpus -v`
Expected: FAIL because artifact generation is missing.

**Step 3: Write minimal implementation**

Add manifest/profile builders and JSON serialization helpers.

**Step 4: Run test to verify it passes**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest tests.test_parallel_corpus -v`
Expected: PASS

### Task 3: Add a rebuild entrypoint

**Files:**
- Create: `T:\mangatl\pipeline\build_parallel_corpus.py`

**Step 1: Write the failing test**

Validate that the script writes artifacts into a deterministic dataset folder.

**Step 2: Run test to verify it fails**

Run the new test.
Expected: FAIL because no entrypoint exists.

**Step 3: Write minimal implementation**

Add a simple script that scans `exemplos/exemploptbr` and `exemplos/exemploen` and writes:
- `manifest.json`
- `quality_profile.json`
- `alignment_profile.json`

**Step 4: Run test to verify it passes**

Run the specific test and inspect the files.

### Task 4: Verify and document

**Files:**
- Modify: `T:\mangatl\context.md`

**Step 1: Run verification**

Run:
- `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest discover -s tests`
- `T:\mangatl\pipeline\venv\Scripts\python.exe T:\mangatl\pipeline\build_parallel_corpus.py`
- `npm run build`

**Step 2: Record outputs**

Document:
- artifact directory
- chapter coverage
- remaining gaps before runtime integration
