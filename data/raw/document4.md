---
title: Brightfield Ingest API
version: 2.3.1
status: stable
last_updated: 2026-02-11
maintainer: platform-team@brightfieldanalytics.example
---

# Brightfield Ingest API

The Ingest API lets you push event data directly into a Brightfield Analytics workspace without
going through the batch upload UI. It's the recommended path for anything higher than ~10k
events/day.

> **Note:** This replaces the old `/v1/events` endpoint, which is deprecated and will be removed
> on 2026-09-01. See [Migration](#migration-from-v1) below.

## Authentication

All requests require a workspace API key passed as a bearer token.

```
Authorization: Bearer bf_live_xxxxxxxxxxxxxxxxxxxx
```

Keys are scoped to a single workspace and can be rotated from **Settings → API Keys**. Test-mode
keys (prefixed `bf_test_`) write to a sandbox dataset that's cleared every 24h.

## Endpoints

### POST /v2/events

Send a single event or a batch of up to 500 events.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `event_name` | string | yes | Name of the event, e.g. `purchase_completed` |
| `timestamp` | string (ISO 8601) | no | Defaults to receipt time if omitted |
| `user_id` | string | yes | Stable identifier for the actor |
| `properties` | object | no | Arbitrary key/value event metadata |

Example:

```json
{
  "event_name": "purchase_completed",
  "timestamp": "2026-02-10T18:32:04Z",
  "user_id": "usr_8f2a1",
  "properties": {
    "amount": 42.50,
    "currency": "USD",
    "items": 3
  }
}
```

Response: `202 Accepted` with an empty body on success. Events are processed asynchronously; use
the [Query API](#query-api-reference-elsewhere) to confirm ingestion (usually within 30s).

### GET /v2/events/schema

Returns the currently inferred schema for your workspace's event stream. Useful for debugging
type mismatches.

no request body needed for this one, just hit it with your API key.

### DELETE /v2/events/{event_id}

Deletes a single event by ID. Only available on Enterprise plans. Deletions are eventually
consistent and can take up to 15 minutes to propagate to downstream dashboards.

## Rate limits

| Plan | Requests/sec | Burst |
|---|---|---|
| Starter | 10 | 20 |
| Growth | 50 | 100 |
| Enterprise | 500 | 1000 |

Exceeding your rate limit returns `429 Too Many Requests` with a `Retry-After` header. We
recommend client-side batching (up to 500 events per call) rather than firing individual requests.

## Error codes

- `400` — malformed JSON or missing required field
- `401` — invalid or expired API key
- `413` — batch exceeds 500 events or 1MB payload
- `422` — event_name doesn't match your workspace's naming policy (if enforced)
- `429` — rate limit exceeded
- `5xx` — internal error, safe to retry with exponential backoff

## SDKs

Official SDKs exist for:
- **Node.js** — `npm install @brightfield/ingest`
- **Python** — `pip install brightfield-ingest`
- Go, Ruby, and Java are community-maintained and not officially supported. Use at your own risk!

## Migration from v1

The old `/v1/events` endpoint used a flat structure with `props` instead of `properties`, and did
not support batching. There's no automated migration tool — you'll need to update your
integration code manually. Reach out in #ingest-api-support if you get stuck.

### Field mapping

| v1 field | v2 field | Notes |
|---|---|---|
| `props` | `properties` | Same shape, renamed |
| `uid` | `user_id` | Renamed for clarity |
| `ts` | `timestamp` | Now requires ISO 8601, v1 accepted unix epoch |
| N/A | (batching) | v2 only — send up to 500 events per request |

---

*Questions? Post in #ingest-api-support or email platform-team@brightfieldanalytics.example.
This doc is auto-generated from the OpenAPI spec for descriptions, but the prose sections above
are hand-maintained and may drift — file a ticket if you spot something stale.*
