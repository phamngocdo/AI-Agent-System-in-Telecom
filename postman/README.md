# Postman API Tests

This directory contains a Postman collection for the FastAPI backend.

Files:

- `TelcoLLM.postman_collection.json`: request collection with test scripts.
- `TelcoLLM.local.postman_environment.json`: local environment variables.
- `sample-upload.md`: markdown file used by the optional file-chat test.

## Run In Postman

1. Start the backend on `http://localhost:8000`.
2. Import both JSON files into Postman.
3. Select the `TelcoLLM Local` environment.
4. Run the `Core Smoke` folder.

The core smoke flow creates a new random test user, logs in, updates the profile, tests session CRUD, validates a missing upload, and deletes the created session.

## Run With Newman

From the repo root:

```sh
newman run postman/TelcoLLM.postman_collection.json \
  -e postman/TelcoLLM.local.postman_environment.json \
  --folder "Core Smoke"
```

## Optional Chat Tests

The chat tests call the configured vLLM service and can be slower. Enable them only when MongoDB, Qdrant, the backend, and vLLM are running:

```sh
newman run postman/TelcoLLM.postman_collection.json \
  -e postman/TelcoLLM.local.postman_environment.json \
  --env-var run_chat_tests=true \
  --folder "Core Smoke" \
  --folder "Chat Optional"
```

To include markdown upload/RAG coverage, also pass:

```sh
--env-var run_file_chat_tests=true --working-dir .
```

The optional requests use `pm.execution.skipRequest()`, so use a recent Postman/Newman version.
