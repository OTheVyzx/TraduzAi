use reqwest::header::RETRY_AFTER;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ANILIST_ENDPOINT: &str = "https://graphql.anilist.co";
const ANILIST_USER_AGENT: &str = "TraduzAI-Studio/0.1 (+local scanlation editor)";
const ANILIST_TIMEOUT_SECONDS: u64 = 10;
const ANILIST_SEARCH_QUERY: &str = r#"
query StudioWorkTrackingSearch($search: String) {
  Media(search: $search, type: MANGA) {
    id
    title { userPreferred english romaji native }
    status
    chapters
    updatedAt
    coverImage { extraLarge large }
    siteUrl
  }
}
"#;
const ANILIST_ID_QUERY: &str = r#"
query StudioWorkTrackingById($id: Int) {
  Media(id: $id, type: MANGA) {
    id
    title { userPreferred english romaji native }
    status
    chapters
    updatedAt
    coverImage { extraLarge large }
    siteUrl
  }
}
"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WorkTrackingSnapshot {
    pub(crate) provider: String,
    pub(crate) provider_id: String,
    pub(crate) title: String,
    pub(crate) status: String,
    pub(crate) remote_chapter_count: Option<f64>,
    pub(crate) cover_url: Option<String>,
    pub(crate) site_url: Option<String>,
    pub(crate) fetched_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrackingError {
    Network(String),
    RateLimited { retry_after_seconds: Option<u64> },
    InvalidResponse(String),
    Provider(String),
    NotFound,
    UnsupportedProvider(String),
    InvalidRequest(String),
}

impl TrackingError {
    pub(crate) fn user_message(&self) -> String {
        match self {
            Self::Network(_) => {
                "Não foi possível conectar ao AniList. Verifique sua conexão e tente novamente.".to_string()
            }
            Self::RateLimited { retry_after_seconds: Some(seconds) } => format!(
                "O AniList limitou temporariamente as consultas. Tente novamente em {seconds} segundos."
            ),
            Self::RateLimited { retry_after_seconds: None } => {
                "O AniList limitou temporariamente as consultas. Tente novamente em alguns instantes.".to_string()
            }
            Self::InvalidResponse(_) => {
                "O AniList retornou uma resposta inválida. Tente novamente mais tarde.".to_string()
            }
            Self::Provider(message) => format!("Não foi possível consultar o AniList: {message}"),
            Self::NotFound => "Nenhuma obra correspondente foi encontrada no AniList.".to_string(),
            Self::UnsupportedProvider(provider) => {
                format!("O provedor {provider} ainda não está disponível no Studio.")
            }
            Self::InvalidRequest(message) => message.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AniListEnvelope {
    data: Option<AniListData>,
    #[serde(default)]
    errors: Vec<AniListGraphqlError>,
}

#[derive(Debug, Deserialize)]
struct AniListData {
    #[serde(rename = "Media")]
    media: Option<AniListMedia>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AniListMedia {
    id: i64,
    title: AniListTitle,
    status: Option<String>,
    chapters: Option<i64>,
    #[serde(rename = "updatedAt")]
    _updated_at: Option<i64>,
    cover_image: Option<AniListCoverImage>,
    site_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AniListTitle {
    user_preferred: Option<String>,
    english: Option<String>,
    romaji: Option<String>,
    native: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AniListCoverImage {
    extra_large: Option<String>,
    large: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AniListGraphqlError {
    message: String,
    status: Option<u16>,
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|item| {
        let trimmed = item.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn preferred_title(title: AniListTitle) -> Option<String> {
    [
        title.english,
        title.user_preferred,
        title.romaji,
        title.native,
    ]
    .into_iter()
    .find_map(non_empty)
}

pub(crate) fn parse_anilist_payload(
    payload: &str,
    fetched_at: &str,
) -> Result<WorkTrackingSnapshot, TrackingError> {
    let envelope: AniListEnvelope = serde_json::from_str(payload)
        .map_err(|error| TrackingError::InvalidResponse(error.to_string()))?;

    if !envelope.errors.is_empty() {
        if envelope
            .errors
            .iter()
            .any(|error| error.status == Some(429))
        {
            return Err(TrackingError::RateLimited {
                retry_after_seconds: None,
            });
        }
        let message = envelope
            .errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(TrackingError::Provider(message));
    }

    let media = envelope
        .data
        .ok_or_else(|| TrackingError::InvalidResponse("campo data ausente".to_string()))?
        .media
        .ok_or(TrackingError::NotFound)?;
    let title = preferred_title(media.title)
        .ok_or_else(|| TrackingError::InvalidResponse("título da obra ausente".to_string()))?;
    let cover_url = media
        .cover_image
        .and_then(|cover| non_empty(cover.extra_large).or_else(|| non_empty(cover.large)));

    Ok(WorkTrackingSnapshot {
        provider: "anilist".to_string(),
        provider_id: media.id.to_string(),
        title,
        status: media.status.unwrap_or_else(|| "UNKNOWN".to_string()),
        remote_chapter_count: media.chapters.map(|chapters| chapters as f64),
        cover_url,
        site_url: non_empty(media.site_url),
        fetched_at: fetched_at.to_string(),
    })
}

pub(crate) fn classify_http_failure(
    status: u16,
    retry_after: Option<&str>,
    body: &str,
) -> TrackingError {
    if status == 429 {
        return TrackingError::RateLimited {
            retry_after_seconds: retry_after.and_then(|value| value.trim().parse::<u64>().ok()),
        };
    }
    let summary = body.trim();
    TrackingError::Provider(if summary.is_empty() {
        format!("HTTP {status}")
    } else {
        format!(
            "HTTP {status}: {}",
            summary.chars().take(160).collect::<String>()
        )
    })
}

pub(crate) fn unix_seconds_to_rfc3339(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_in_day = seconds % 86_400;
    let hour = seconds_in_day / 3_600;
    let minute = (seconds_in_day % 3_600) / 60;
    let second = seconds_in_day % 60;

    let shifted_days = days + 719_468;
    let era = shifted_days.div_euclid(146_097);
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn current_fetch_timestamp() -> Result<String, TrackingError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            TrackingError::InvalidResponse(format!("relógio do sistema inválido: {error}"))
        })?
        .as_secs();
    Ok(unix_seconds_to_rfc3339(seconds))
}

pub(crate) fn anilist_search_payload(query: &str) -> Value {
    json!({ "query": ANILIST_SEARCH_QUERY, "variables": { "search": query } })
}

pub(crate) fn anilist_id_payload(id: i64) -> Value {
    json!({ "query": ANILIST_ID_QUERY, "variables": { "id": id } })
}

async fn request_anilist(request_payload: Value) -> Result<WorkTrackingSnapshot, TrackingError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(ANILIST_TIMEOUT_SECONDS))
        .user_agent(ANILIST_USER_AGENT)
        .build()
        .map_err(|error| TrackingError::Network(error.to_string()))?;
    let response = client
        .post(ANILIST_ENDPOINT)
        .json(&request_payload)
        .send()
        .await
        .map_err(|error| TrackingError::Network(error.to_string()))?;
    let status = response.status();
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let payload = response
        .text()
        .await
        .map_err(|error| TrackingError::InvalidResponse(error.to_string()))?;
    if !status.is_success() {
        return Err(classify_http_failure(
            status.as_u16(),
            retry_after.as_deref(),
            &payload,
        ));
    }
    let fetched_at = current_fetch_timestamp()?;
    parse_anilist_payload(&payload, &fetched_at)
}

#[tauri::command]
pub(crate) async fn studio_search_tracking_works(
    query: String,
    provider: String,
) -> Result<Vec<WorkTrackingSnapshot>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err(TrackingError::InvalidRequest(
            "Informe o nome da obra para pesquisar.".to_string(),
        )
        .user_message());
    }
    if !provider.eq_ignore_ascii_case("anilist") {
        return Err(TrackingError::UnsupportedProvider(provider).user_message());
    }
    request_anilist(anilist_search_payload(query))
        .await
        .map(|snapshot| vec![snapshot])
        .map_err(|error| error.user_message())
}

#[tauri::command]
pub(crate) async fn studio_sync_tracking_work(
    anilist_id: Option<i64>,
    manga_dex_id: Option<String>,
) -> Result<Vec<WorkTrackingSnapshot>, String> {
    if let Some(id) = anilist_id.filter(|id| *id > 0) {
        return request_anilist(anilist_id_payload(id))
            .await
            .map(|snapshot| vec![snapshot])
            .map_err(|error| error.user_message());
    }
    if manga_dex_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty())
    {
        return Err(TrackingError::UnsupportedProvider("MangaDex".to_string()).user_message());
    }
    Err(TrackingError::InvalidRequest(
        "Vincule a obra ao AniList ou MangaDex antes de atualizar.".to_string(),
    )
    .user_message())
}

#[cfg(test)]
mod tests {
    use super::{
        anilist_id_payload, anilist_search_payload, classify_http_failure, parse_anilist_payload,
        unix_seconds_to_rfc3339, TrackingError,
    };

    #[test]
    fn parses_anilist_fixture_into_the_normalized_snapshot() {
        let snapshot = parse_anilist_payload(
            include_str!("fixtures/anilist_media.json"),
            "2026-07-22T12:00:00Z",
        )
        .expect("fixture should parse");

        assert_eq!(snapshot.provider, "anilist");
        assert_eq!(snapshot.provider_id, "105398");
        assert_eq!(snapshot.title, "Solo Leveling");
        assert_eq!(snapshot.status, "FINISHED");
        assert_eq!(snapshot.remote_chapter_count, Some(200.0));
        assert_eq!(
            snapshot.cover_url.as_deref(),
            Some("https://s4.anilist.co/file/anilistcdn/media/manga/cover/large/bx105398.jpg")
        );
        assert_eq!(
            snapshot.site_url.as_deref(),
            Some("https://anilist.co/manga/105398")
        );
        assert_eq!(snapshot.fetched_at, "2026-07-22T12:00:00Z");
    }

    #[test]
    fn distinguishes_network_rate_limit_and_invalid_response_errors() {
        let network = TrackingError::Network("offline".to_string());
        let rate_limit = classify_http_failure(429, Some("42"), "");
        let invalid = parse_anilist_payload("not-json", "2026-07-22T12:00:00Z")
            .expect_err("invalid JSON must fail");

        assert!(matches!(network, TrackingError::Network(_)));
        assert!(matches!(
            rate_limit,
            TrackingError::RateLimited {
                retry_after_seconds: Some(42)
            }
        ));
        assert!(matches!(invalid, TrackingError::InvalidResponse(_)));
        assert!(TrackingError::Network("offline".to_string())
            .user_message()
            .contains("conectar ao AniList"));
        assert!(classify_http_failure(429, Some("42"), "")
            .user_message()
            .contains("42 segundos"));
    }

    #[test]
    fn recognizes_graphql_rate_limit_errors_even_on_a_json_response() {
        let error = parse_anilist_payload(
            r#"{"data":null,"errors":[{"message":"Too Many Requests.","status":429}]}"#,
            "2026-07-22T12:00:00Z",
        )
        .expect_err("GraphQL 429 must fail");

        assert!(matches!(error, TrackingError::RateLimited { .. }));
    }

    #[test]
    fn formats_fetch_time_as_utc_rfc3339() {
        assert_eq!(unix_seconds_to_rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(
            unix_seconds_to_rfc3339(1_753_185_600),
            "2025-07-22T12:00:00Z"
        );
    }

    #[test]
    fn keeps_search_and_id_queries_separate_for_anilist() {
        let search = anilist_search_payload("Solo Leveling");
        let sync = anilist_id_payload(105398);

        assert!(search["query"]
            .as_str()
            .unwrap()
            .contains("Media(search: $search, type: MANGA)"));
        assert!(!search["query"].as_str().unwrap().contains("$id"));
        assert_eq!(
            search["variables"],
            serde_json::json!({ "search": "Solo Leveling" })
        );
        assert!(sync["query"]
            .as_str()
            .unwrap()
            .contains("Media(id: $id, type: MANGA)"));
        assert!(!sync["query"].as_str().unwrap().contains("$search"));
        assert_eq!(sync["variables"], serde_json::json!({ "id": 105398 }));
    }

    #[test]
    fn prefers_the_english_title_for_the_en_to_pt_br_library() {
        let snapshot = parse_anilist_payload(
            r#"{
              "data": {
                "Media": {
                  "id": 105398,
                  "title": {
                    "userPreferred": "Na Honjaman Level Up",
                    "english": "Solo Leveling",
                    "romaji": "Ore dake Level Up na Ken",
                    "native": "나 혼자만 레벨업"
                  },
                  "status": "FINISHED",
                  "chapters": 201,
                  "updatedAt": 1753185600,
                  "coverImage": null,
                  "siteUrl": "https://anilist.co/manga/105398"
                }
              }
            }"#,
            "2026-07-22T12:00:00Z",
        )
        .expect("payload should parse");

        assert_eq!(snapshot.title, "Solo Leveling");
    }
}
