use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const PROTOCOL_VERSION: u16 = 1;
#[cfg(test)]
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
#[cfg(test)]
const MAX_DEADLINE_MS: u64 = 300_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProtocolRequest {
    #[serde(rename = "v")]
    pub(crate) version: u16,
    pub(crate) id: String,
    pub(crate) method: String,
    #[serde(rename = "deadlineMs")]
    pub(crate) deadline_ms: u64,
    pub(crate) params: Value,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ProtocolError {
    pub(crate) code: &'static str,
    pub(crate) message: String,
    pub(crate) retryable: bool,
}

#[cfg(test)]
impl ProtocolError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "INVALID_REQUEST",
            message: message.into(),
            retryable: false,
        }
    }
}

#[cfg(test)]
pub(crate) fn parse_request_line(line: &str) -> Result<ProtocolRequest, ProtocolError> {
    if line.len() > MAX_REQUEST_BYTES {
        return Err(ProtocolError::invalid("Request exceeds the protocol limit"));
    }
    let request: ProtocolRequest = serde_json::from_str(line)
        .map_err(|error| ProtocolError::invalid(format!("Invalid JSON request: {error}")))?;
    if request.version != PROTOCOL_VERSION {
        return Err(ProtocolError {
            code: "PROTOCOL_INCOMPATIBLE",
            message: format!(
                "Protocol {} is not supported; expected {}",
                request.version, PROTOCOL_VERSION
            ),
            retryable: false,
        });
    }
    if request.id.trim().is_empty() || request.method.trim().is_empty() {
        return Err(ProtocolError::invalid("Request id and method are required"));
    }
    if request.deadline_ms == 0 || request.deadline_ms > MAX_DEADLINE_MS {
        return Err(ProtocolError::invalid(
            "Request deadline is outside the allowed range",
        ));
    }
    if !request.params.is_object() {
        return Err(ProtocolError::invalid("Request params must be an object"));
    }
    Ok(request)
}
