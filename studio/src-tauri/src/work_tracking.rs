use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ANILIST_ENDPOINT: &str = "https://graphql.anilist.co";
const ANILIST_USER_AGENT: &str = "TraduzAI-Studio/0.1 (+local scanlation editor)";
const ANILIST_TIMEOUT_SECONDS: u64 = 10;
const MANGADEX_ENDPOINT: &str = "https://api.mangadex.org";
const MANGADEX_COVER_ENDPOINT: &str = "https://uploads.mangadex.org/covers";
const REQUEST_TIMEOUT_SECONDS: u64 = 12;
const MAX_REQUEST_ATTEMPTS: usize = 3;
const MANGADEX_PAGE_LIMIT: usize = 500;
const MAX_MANGADEX_FEED_PAGES: usize = 20;
const MANGADEX_RETRY_AFTER: &str = "x-ratelimit-retry-after";
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
    pub(crate) latest_chapter: Option<String>,
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
                "Não foi possível conectar ao provedor. Os dados salvos continuam disponíveis.".to_string()
            }
            Self::RateLimited { retry_after_seconds: Some(seconds) } => format!(
                "O provedor limitou temporariamente as consultas. Tente novamente em {seconds} segundos."
            ),
            Self::RateLimited { retry_after_seconds: None } => {
                "O provedor limitou temporariamente as consultas. Tente novamente em alguns instantes.".to_string()
            }
            Self::InvalidResponse(_) => {
                "O provedor retornou uma resposta inválida. Tente novamente mais tarde.".to_string()
            }
            Self::Provider(message) => format!("Não foi possível consultar o provedor: {message}"),
            Self::NotFound => "Nenhuma obra correspondente foi encontrada.".to_string(),
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
        latest_chapter: media.chapters.map(|chapters| chapters.to_string()),
        cover_url,
        site_url: non_empty(media.site_url),
        fetched_at: fetched_at.to_string(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MangaDexFeedSummary {
    pub(crate) chapter_labels: Vec<String>,
    pub(crate) latest_chapter: Option<String>,
    pub(crate) chapter_count: usize,
}

fn value_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn mangadex_title(attributes: &Value) -> Option<String> {
    let titles = attributes.get("title")?.as_object()?;
    titles
        .get("en")
        .and_then(Value::as_str)
        .or_else(|| titles.values().find_map(Value::as_str))
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

fn mangadex_cover_url(manga_id: &str, relationships: Option<&Value>) -> Option<String> {
    let file_name = relationships?
        .as_array()?
        .iter()
        .find(|relationship| relationship.get("type").and_then(Value::as_str) == Some("cover_art"))?
        .get("attributes")?
        .get("fileName")?
        .as_str()?
        .trim();
    if file_name.is_empty() {
        None
    } else {
        Some(format!(
            "{MANGADEX_COVER_ENDPOINT}/{manga_id}/{file_name}.256.jpg"
        ))
    }
}

fn parse_mangadex_manga(
    value: &Value,
    fetched_at: &str,
) -> Result<WorkTrackingSnapshot, TrackingError> {
    let manga_id = value_string(value.get("id"))
        .ok_or_else(|| TrackingError::InvalidResponse("ID MangaDex ausente".to_string()))?;
    let attributes = value
        .get("attributes")
        .ok_or_else(|| TrackingError::InvalidResponse("atributos MangaDex ausentes".to_string()))?;
    let title = mangadex_title(attributes)
        .ok_or_else(|| TrackingError::InvalidResponse("título MangaDex ausente".to_string()))?;

    Ok(WorkTrackingSnapshot {
        provider: "mangadex".to_string(),
        provider_id: manga_id.clone(),
        title,
        status: value_string(attributes.get("status")).unwrap_or_else(|| "unknown".to_string()),
        remote_chapter_count: None,
        latest_chapter: None,
        cover_url: mangadex_cover_url(&manga_id, value.get("relationships")),
        site_url: Some(format!("https://mangadex.org/title/{manga_id}")),
        fetched_at: fetched_at.to_string(),
    })
}

pub(crate) fn parse_mangadex_manga_payload(
    payload: &str,
    fetched_at: &str,
) -> Result<Vec<WorkTrackingSnapshot>, TrackingError> {
    let root: Value = serde_json::from_str(payload)
        .map_err(|error| TrackingError::InvalidResponse(error.to_string()))?;
    if root.get("result").and_then(Value::as_str) != Some("ok") {
        return Err(TrackingError::Provider(
            value_string(root.get("result"))
                .unwrap_or_else(|| "resposta MangaDex sem sucesso".to_string()),
        ));
    }
    let data = root
        .get("data")
        .ok_or_else(|| TrackingError::InvalidResponse("campo data ausente".to_string()))?;
    let items: Vec<&Value> = match data {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![data],
        _ => {
            return Err(TrackingError::InvalidResponse(
                "campo data inválido".to_string(),
            ))
        }
    };
    if items.is_empty() {
        return Err(TrackingError::NotFound);
    }
    items
        .into_iter()
        .map(|item| parse_mangadex_manga(item, fetched_at))
        .collect()
}

fn chapter_label(attributes: &Value) -> String {
    value_string(attributes.get("chapter"))
        .or_else(|| value_string(attributes.get("title")))
        .unwrap_or_else(|| "Especial".to_string())
}

fn compare_chapter_labels(left: &str, right: &str) -> Ordering {
    let left_number = left.replace(',', ".").parse::<f64>().ok();
    let right_number = right.replace(',', ".").parse::<f64>().ok();
    match (left_number, right_number) {
        (Some(left), Some(right)) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left.to_lowercase().cmp(&right.to_lowercase()),
    }
}

pub(crate) fn parse_mangadex_feed_payload(
    payload: &str,
    language: &str,
) -> Result<MangaDexFeedSummary, TrackingError> {
    let root: Value = serde_json::from_str(payload)
        .map_err(|error| TrackingError::InvalidResponse(error.to_string()))?;
    let items = root
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| TrackingError::InvalidResponse("feed MangaDex sem data".to_string()))?;
    let mut chapters: HashMap<String, (String, String)> = HashMap::new();

    for item in items {
        let Some(attributes) = item.get("attributes") else {
            continue;
        };
        if value_string(attributes.get("translatedLanguage")).as_deref() != Some(language) {
            continue;
        }
        let label = chapter_label(attributes);
        let key = label.trim().to_lowercase();
        let published_at = value_string(attributes.get("publishAt"))
            .or_else(|| value_string(attributes.get("readableAt")))
            .unwrap_or_default();
        let should_replace = chapters
            .get(&key)
            .is_none_or(|(_, existing_date)| published_at > *existing_date);
        if should_replace {
            chapters.insert(key, (label, published_at));
        }
    }

    let latest_chapter = chapters
        .values()
        .max_by(|(_, left_date), (_, right_date)| left_date.cmp(right_date))
        .map(|(label, _)| label.clone());
    let mut chapter_labels = chapters
        .into_values()
        .map(|(label, _)| label)
        .collect::<Vec<_>>();
    chapter_labels.sort_by(|left, right| compare_chapter_labels(left, right));

    Ok(MangaDexFeedSummary {
        chapter_count: chapter_labels.len(),
        chapter_labels,
        latest_chapter,
    })
}

pub(crate) fn retry_delay(
    attempt: usize,
    status: u16,
    retry_after: Option<&str>,
) -> Option<Duration> {
    if attempt + 1 >= MAX_REQUEST_ATTEMPTS || !matches!(status, 429 | 500 | 502 | 503 | 504) {
        return None;
    }
    if status == 429 {
        if let Some(seconds) = retry_after.and_then(|value| value.trim().parse::<u64>().ok()) {
            return Some(Duration::from_secs(seconds.min(30)));
        }
    }
    Some(Duration::from_millis(250 * (1_u64 << attempt.min(4))))
}

fn retry_after_header(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
    {
        return Some(value.to_string());
    }
    let reset_at = headers
        .get(MANGADEX_RETRY_AFTER)
        .and_then(|value| value.to_str().ok())?
        .trim()
        .parse::<u64>()
        .ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(reset_at.saturating_sub(now).max(1).to_string())
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

fn tracking_client() -> Result<reqwest::Client, TrackingError> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECONDS))
        .user_agent(ANILIST_USER_AGENT)
        .build()
        .map_err(|error| TrackingError::Network(error.to_string()))
}

async fn send_with_retry<F>(mut build_request: F) -> Result<String, TrackingError>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    for attempt in 0..MAX_REQUEST_ATTEMPTS {
        let response = match build_request().send().await {
            Ok(response) => response,
            Err(error) if attempt + 1 < MAX_REQUEST_ATTEMPTS => {
                tokio::time::sleep(Duration::from_millis(250 * (1_u64 << attempt))).await;
                let _ = error;
                continue;
            }
            Err(error) => return Err(TrackingError::Network(error.to_string())),
        };
        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        if let Some(delay) = retry_delay(attempt, status.as_u16(), retry_after.as_deref()) {
            tokio::time::sleep(delay).await;
            continue;
        }
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
        return Ok(payload);
    }
    Err(TrackingError::Network("tentativas esgotadas".to_string()))
}

async fn request_anilist(request_payload: Value) -> Result<WorkTrackingSnapshot, TrackingError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(ANILIST_TIMEOUT_SECONDS))
        .user_agent(ANILIST_USER_AGENT)
        .build()
        .map_err(|error| TrackingError::Network(error.to_string()))?;
    let payload = send_with_retry(|| client.post(ANILIST_ENDPOINT).json(&request_payload)).await?;
    let fetched_at = current_fetch_timestamp()?;
    parse_anilist_payload(&payload, &fetched_at)
}

fn looks_like_mangadex_id(value: &str) -> bool {
    value.len() == 36
        && value.chars().enumerate().all(|(index, character)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                character == '-'
            } else {
                character.is_ascii_hexdigit()
            }
        })
}

async fn request_mangadex_search(query: &str) -> Result<Vec<WorkTrackingSnapshot>, TrackingError> {
    let client = tracking_client()?;
    let payload = if looks_like_mangadex_id(query) {
        let url = format!("{MANGADEX_ENDPOINT}/manga/{query}");
        send_with_retry(|| client.get(&url).query(&[("includes[]", "cover_art")])).await?
    } else {
        let url = format!("{MANGADEX_ENDPOINT}/manga");
        send_with_retry(|| {
            client.get(&url).query(&[
                ("title", query),
                ("limit", "10"),
                ("includes[]", "cover_art"),
            ])
        })
        .await?
    };
    parse_mangadex_manga_payload(&payload, &current_fetch_timestamp()?)
}

async fn request_mangadex_by_id(
    manga_id: &str,
    language: &str,
) -> Result<WorkTrackingSnapshot, TrackingError> {
    let mut snapshots = request_mangadex_search(manga_id).await?;
    let mut snapshot = snapshots.pop().ok_or(TrackingError::NotFound)?;
    let client = tracking_client()?;
    let url = format!("{MANGADEX_ENDPOINT}/manga/{manga_id}/feed");
    let mut chapters = Vec::new();
    let mut offset = 0_usize;
    for page in 0..MAX_MANGADEX_FEED_PAGES {
        let offset_value = offset.to_string();
        let limit_value = MANGADEX_PAGE_LIMIT.to_string();
        let payload = send_with_retry(|| {
            client.get(&url).query(&[
                ("translatedLanguage[]", language),
                ("order[publishAt]", "desc"),
                ("limit", limit_value.as_str()),
                ("offset", offset_value.as_str()),
                ("includeFutureUpdates", "0"),
            ])
        })
        .await?;
        let page_payload: Value = serde_json::from_str(&payload)
            .map_err(|error| TrackingError::InvalidResponse(error.to_string()))?;
        let page_items = page_payload
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| TrackingError::InvalidResponse("feed MangaDex sem data".to_string()))?;
        let item_count = page_items.len();
        chapters.extend(page_items.iter().cloned());
        let total = page_payload
            .get("total")
            .and_then(Value::as_u64)
            .unwrap_or(item_count as u64) as usize;
        offset += item_count;
        if item_count == 0 || offset >= total {
            break;
        }
        if page + 1 < MAX_MANGADEX_FEED_PAGES {
            tokio::time::sleep(Duration::from_millis(220)).await;
        }
    }
    let payload = json!({ "result": "ok", "data": chapters }).to_string();
    let feed = parse_mangadex_feed_payload(&payload, language)?;
    snapshot.remote_chapter_count = Some(feed.chapter_count as f64);
    snapshot.latest_chapter = feed.latest_chapter;
    Ok(snapshot)
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
    if provider.eq_ignore_ascii_case("anilist") {
        return request_anilist(anilist_search_payload(query))
            .await
            .map(|snapshot| vec![snapshot])
            .map_err(|error| error.user_message());
    }
    if provider.eq_ignore_ascii_case("mangadex") {
        return request_mangadex_search(query)
            .await
            .map_err(|error| error.user_message());
    }
    Err(TrackingError::UnsupportedProvider(provider).user_message())
}

#[tauri::command]
pub(crate) async fn studio_sync_tracking_work(
    anilist_id: Option<i64>,
    manga_dex_id: Option<String>,
    tracking_language: Option<String>,
) -> Result<Vec<WorkTrackingSnapshot>, String> {
    let mut snapshots = Vec::new();
    let mut errors = Vec::new();
    if let Some(id) = anilist_id.filter(|id| *id > 0) {
        match request_anilist(anilist_id_payload(id)).await {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(error) => errors.push(error.user_message()),
        }
    }
    if let Some(manga_id) = manga_dex_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        let language = tracking_language
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("en");
        match request_mangadex_by_id(manga_id, language).await {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(error) => errors.push(error.user_message()),
        }
    }
    if !snapshots.is_empty() {
        return Ok(snapshots);
    }
    if let Some(error) = errors.into_iter().next() {
        return Err(error);
    }
    Err(TrackingError::InvalidRequest(
        "Vincule a obra ao AniList ou MangaDex antes de atualizar.".to_string(),
    )
    .user_message())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{
        anilist_id_payload, anilist_search_payload, classify_http_failure, parse_anilist_payload,
        parse_mangadex_feed_payload, parse_mangadex_manga_payload, retry_after_header, retry_delay,
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
            .contains("conectar ao provedor"));
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

    #[test]
    fn parses_mangadex_metadata_into_the_normalized_snapshot() {
        let snapshots = parse_mangadex_manga_payload(
            r#"{
              "result":"ok",
              "data": [{
                "id":"ade0306c-f4b6-4890-9edb-1ddf04df2039",
                "type":"manga",
                "attributes": {
                  "title":{"en":"Solo Leveling"},
                  "altTitles":[{"ko":"Na Honjaman Level Up"}],
                  "status":"hiatus"
                },
                "relationships":[{
                  "id":"cover-id",
                  "type":"cover_art",
                  "attributes":{"fileName":"cover.jpg"}
                }]
              }]
            }"#,
            "2026-07-22T12:00:00Z",
        )
        .expect("MangaDex metadata should parse");

        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].provider, "mangadex");
        assert_eq!(snapshots[0].title, "Solo Leveling");
        assert_eq!(snapshots[0].status, "hiatus");
        assert_eq!(snapshots[0].latest_chapter, None);
        assert!(snapshots[0]
            .cover_url
            .as_deref()
            .unwrap()
            .contains("/covers/ade0306c-f4b6-4890-9edb-1ddf04df2039/cover.jpg"));
    }

    #[test]
    fn filters_language_and_deduplicates_scan_groups_by_latest_date() {
        let feed = parse_mangadex_feed_payload(include_str!("fixtures/mangadex_feed.json"), "en")
            .expect("feed should parse");

        assert_eq!(feed.chapter_labels, vec!["10", "10.5", "Extra"]);
        assert_eq!(feed.latest_chapter.as_deref(), Some("10.5"));
        assert_eq!(feed.chapter_count, 3);
    }

    #[test]
    fn bounds_transient_retry_and_respects_retry_after() {
        assert_eq!(retry_delay(0, 429, Some("3")), Some(Duration::from_secs(3)));
        assert_eq!(retry_delay(1, 503, None), Some(Duration::from_millis(500)));
        assert_eq!(retry_delay(2, 503, None), None);
        assert_eq!(retry_delay(0, 404, None), None);

        let reset_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 4;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-ratelimit-retry-after",
            reset_at.to_string().parse().unwrap(),
        );
        let delay = retry_after_header(&headers)
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert!((3..=4).contains(&delay));
    }
}
