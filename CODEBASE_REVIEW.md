# Codebase Walkthrough

## High-level structure

This repository contains multiple entry points and **at least two partially divergent implementations** of an Aadhaar + face-recognition system:

1. **Streamlit path** (`app.py`) using `AadhaarFaceSystem` from `aadhaar_system.py`.
2. **CLI path** (`main.py`) also using `AadhaarFaceSystem`.
3. **Flask path** (`web_interface.py`) that expects a richer `AadhaarFaceSystem` API than currently provided by `aadhaar_system.py`.
4. **Alternative architecture draft** in `system_architecture.py` that defines another `AadhaarFaceSystem` with a different backend and method signatures.

Because these paths rely on inconsistent method names and schemas, the current app variants are not plug-and-play interchangeable.

## What each core file does

- `aadhaar_system.py`:
  - Creates `persons` table with columns: `aadhaar`, `name`, `face_encoding`, `face_path`, `aadhaar_path`, `created_at`.
  - Uses EasyOCR (`extract_aadhaar`) to parse a 12-digit number from card image text.
  - Uses `face_recognition` to encode, store, and match faces.
  - Exposes `register(face_img, aadhaar_img)` and `recognize(frame)`.

- `app.py` (Streamlit):
  - Captures webcam frames via `streamlit-webrtc`.
  - In Register mode, stores snapshots in session and calls `system.register(...)`.
  - In Recognize mode, calls `system.recognize(...)` every frame and overlays labels.

- `main.py` (CLI):
  - Minimal console menu calling `register_live()` and `recognize_live()`.
  - These methods are **not present** in the current `aadhaar_system.py`, so this path appears stale.

- `web_interface.py` (Flask):
  - Expects methods such as `register_person`, `recognize_face`, and `get_person_details`.
  - Instantiates `AadhaarFaceSystem(debug=True)` though `aadhaar_system.py` constructor does not accept `debug`.
  - Contains useful image encode/decode helpers and API routes, but is presently coupled to a different class contract.

- `system_architecture.py`:
  - A separate, cleaner class prototype using InsightFace and a different DB schema (`aadhaar_number`, embedding size assumptions, etc.).
  - Method names differ (`register_person_from_pil`, `recognize_face_from_pil`).

## Main mismatches identified

1. **Constructor mismatch**
   - `web_interface.py` uses `AadhaarFaceSystem(debug=True)`.
   - `aadhaar_system.py` constructor accepts no `debug` arg.

2. **Method mismatch (Flask vs core class)**
   - Flask expects: `register_person`, `recognize_face`, `get_person_details`.
   - Core class provides: `register`, `recognize`, `extract_aadhaar`.

3. **Method mismatch (CLI vs core class)**
   - CLI expects: `register_live`, `recognize_live`.
   - Core class does not provide them.

4. **DB schema mismatch**
   - `aadhaar_system.py` uses primary key `aadhaar`.
   - Other modules query/update by `aadhaar_number` and expect columns like `registered_face_path`, `aadhaar_photo_path`.

5. **Multiple competing implementations**
   - `aadhaar_system.py` and `system_architecture.py` both define a similarly named system class but with different APIs and ML stacks.

## Recommended consolidation path

1. Pick **one** canonical backend class (`AadhaarFaceSystem`) and freeze its public API.
2. Align all entry points (Streamlit/Flask/CLI) to that same API.
3. Unify on a single `persons` schema (`aadhaar_number` recommended for clarity).
4. Keep adapter shims temporarily if migration is needed, then remove stale paths.
5. Add a smoke test script that verifies:
   - system initialization,
   - DB migration success,
   - one registration flow,
   - one recognition flow.

## Quick status summary

- **Most coherent runnable path right now**: `app.py` + current `aadhaar_system.py` methods.
- **Likely broken without refactor**: `main.py`, `web_interface.py` against current `aadhaar_system.py`.
