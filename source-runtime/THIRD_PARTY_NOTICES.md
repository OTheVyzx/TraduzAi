# Third-party notices for the Source Runtime

This inventory is the compliance baseline. Exact copied files, local modifications, binary hashes and transitive dependencies must be generated and verified before distribution.

## Suwayomi-Server compatibility runtime and AndroidCompat

- Upstream: https://github.com/Suwayomi/Suwayomi-Server
- Pinned revision: `ac3dd314dba275fdadc4a3208002a9fa42135bf2`
- License: Mozilla Public License 2.0
- Usage: the private worker classpath is built from this exact revision; the TraduzAI broker does not start Suwayomi's HTTP or GraphQL server.
- Policy: MPL binaries and any copied or modified MPL files retain notices and ship with the exact corresponding source revision.

## Mihon compatibility API

- Upstream: https://github.com/mihonapp/mihon
- Pinned revision: `056d9baeeba6b7f53c0638b3d30da72e71c159fa`
- License: Apache License 2.0
- Policy: retain notices, copyright statements and modification notices.

## Keiyoushi fixture and catalog format references

- Upstream: https://github.com/keiyoushi/extensions-source
- Pinned revision: `2063590a39622a68075a4cb8834edec8b11d0986`
- License: Apache License 2.0
- Policy: production repositories and extensions are never bundled; test fixtures must be original or license-compatible.

## Eclipse Temurin

- Release: `jdk-21.0.12+1`, Windows x64 HotSpot
- Policy: the final runtime image, source offer, license files and cryptographic hashes are produced by the release pipeline.

This file is not legal advice. Distribution remains blocked until the generated SBOM and corresponding-source package have been reviewed.
