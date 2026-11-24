//! Native Sign in with ChatGPT (SIWC) inference and account model discovery.
//! This transport never uses API keys, Codex backend endpoints, or stored responses.

use std::{collections::HashSet, time::Duration};

use async_trait::async_trait;
use edgequake_llm::{
    error::{LlmError, Result},
    traits::{
        ChatMessage, ChatRole, CompletionOptions, FunctionCall, LLMProvider, LLMResponse,
        StreamChunk, StreamUsage, ToolCall, ToolChoice, ToolDefinition,
    },
};
use futures::{Stream, StreamExt, stream, stream::BoxStream};
use reqwest::{Client, Response, StatusCode, header};
use serde_json::{Value, json};

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
const MODELS_URL: &str = "https://api.openai.com/v1/models";
const NAMESPACE: &str = "edgecrab";
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 64 * 1024;
const MAX_CATALOG_BYTES: usize = 4 * 1024 * 1024;

/// A model advertised for display by the currently selected ChatGPT account.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubscriptionModel {
    pub id: String,
    pub label: String,
}

/// Fetch the active account's displayable models in the order OpenAI returned them.
/// An empty list is valid; a failed fetch is an error, never a cached catalog.
pub async fn list_models() -> Result<Vec<SubscriptionModel>> {
    let http = client()?;
    let token = access_token().await?;
    let mut response = send(
        http.get(MODELS_URL)
            .header(header::ACCEPT, "application/json")
            .bearer_auth(token),
    )
    .await?;
    let diagnostic = Diagnostic::from_response(&response);
    let body = bounded_body(&mut response, MAX_CATALOG_BYTES)
        .await
        .map_err(|()| diagnostic.failure("could not read model catalog", None))?;
    parse_models(&body).map_err(|()| diagnostic.failure("invalid model catalog", None))
}

fn parse_models(body: &[u8]) -> std::result::Result<Vec<SubscriptionModel>, ()> {
    let value: Value = serde_json::from_slice(body).map_err(|_| ())?;
    value["models"]
        .as_array()
        .ok_or(())?
        .iter()
        .filter(|model| model["visibility"].as_str() == Some("list"))
        .map(|model| {
            Ok(SubscriptionModel {
                id: nonempty(&model["slug"]).ok_or(())?.to_owned(),
                label: nonempty(&model["display_name"]).ok_or(())?.to_owned(),
            })
        })
        .collect()
}

/// ChatGPT subscription transport for the selected model slug. The context limit
/// is supplied by the caller for local budgeting, not sent over the wire.
pub struct ChatGptSubscriptionProvider {
    http: Client,
    model: String,
    max_context_length: usize,
}

impl ChatGptSubscriptionProvider {
    pub fn new(model: impl Into<String>, max_context_length: usize) -> Result<Self> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(invalid("a ChatGPT subscription model must be selected"));
        }
        Ok(Self {
            http: client()?,
            model,
            max_context_length,
        })
    }

    async fn events(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<Event>>> {
        let request = build_request(&self.model, messages, tools, choice, options)?;
        let token = access_token().await?;
        let response = send(
            self.http
                .post(RESPONSES_URL)
                .header(header::ACCEPT, "text/event-stream")
                .bearer_auth(token)
                .json(&request.body),
        )
        .await?;
        let diagnostic = Diagnostic::from_response(&response);
        if !stream_content_type_supported(response.headers().get(header::CONTENT_TYPE)) {
            return Err(diagnostic.failure("expected an event stream", None));
        }
        Ok(decode_stream(
            response.bytes_stream().boxed(),
            diagnostic,
            self.model.clone(),
            request.callable,
        ))
    }
}

fn stream_content_type_supported(value: Option<&header::HeaderValue>) -> bool {
    // The subscription endpoint can omit Content-Type on a valid SSE response.
    // The bounded parser still requires a valid response.completed event.
    value.is_none_or(|value| {
        value
            .to_str()
            .ok()
            .and_then(|value| value.split(';').next())
            .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"))
    })
}

#[async_trait]
impl LLMProvider for ChatGptSubscriptionProvider {
    fn name(&self) -> &str {
        "chatgpt-pro"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.chat(&[ChatMessage::user(prompt)], None).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        self.chat(&[ChatMessage::user(prompt)], Some(options)).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.chat_with_tools(messages, &[], None, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let mut events = self.events(messages, tools, tool_choice, options).await?;
        while let Some(event) = events.next().await {
            if let Event::Completed { response, .. } = event? {
                return Ok(response);
            }
        }
        // decode_stream reports premature EOF itself, including the request ID.
        Err(LlmError::ApiError(
            "ChatGPT subscription: stream ended without response.completed".into(),
        ))
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        Ok(self
            .chat_with_tools_stream(&[ChatMessage::user(prompt)], &[], None, None)
            .await?
            .filter_map(|chunk| async {
                match chunk {
                    Ok(StreamChunk::Content(content)) => Some(Ok(content)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .boxed())
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let events = self.events(messages, tools, tool_choice, options).await?;
        Ok(stream::once(async {
            Ok(StreamChunk::Connected {
                phase: "sse_open".into(),
            })
        })
        .chain(to_chunks(events))
        .boxed())
    }
}

fn client() -> Result<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(120))
        .build()
        .map_err(|_| {
            LlmError::ConfigError("could not initialize ChatGPT subscription HTTP client".into())
        })
}

async fn access_token() -> Result<String> {
    // The OAuth boundary's error chain may contain private credential/storage data.
    crate::oauth::chatgpt::access_token().await.map_err(|_| {
        LlmError::AuthError(
            "could not obtain a ChatGPT subscription token; check the selected account and its plan permission, or run `edgecrab auth login chatgpt-pro` if signed out".into(),
        )
    })
}

async fn send(request: reqwest::RequestBuilder) -> Result<Response> {
    let mut response = request.send().await.map_err(|error| {
        let message = if error.is_timeout() {
            "ChatGPT subscription HTTP request timed out"
        } else {
            "ChatGPT subscription HTTP request could not be sent"
        };
        LlmError::NetworkError(message.into())
    })?;
    if response.status().is_success() {
        return Ok(response);
    }
    let diagnostic = Diagnostic::from_response(&response);
    let body: Option<Value> = bounded_body(&mut response, MAX_ERROR_BYTES)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let error = body.as_ref().and_then(|body| body.get("error"));
    let summary = match response.status() {
        StatusCode::UNAUTHORIZED => {
            "request not authorized; check the selected ChatGPT account and its plan permission"
        }
        StatusCode::FORBIDDEN => "request forbidden by permission or policy; check account access",
        StatusCode::NOT_FOUND => {
            "request target not found; for inference, choose a slug from the active ChatGPT account's model list"
        }
        StatusCode::TOO_MANY_REQUESTS => {
            "request limited; check your ChatGPT plan limits and try again later"
        }
        StatusCode::SERVICE_UNAVAILABLE => "service unavailable; try again later",
        _ => "HTTP request rejected",
    };
    Err(diagnostic.failure(summary, error))
}

async fn bounded_body(response: &mut Response, limit: usize) -> std::result::Result<Vec<u8>, ()> {
    let mut result = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
        if chunk.len() > limit - result.len() {
            return Err(());
        }
        result.extend_from_slice(&chunk);
    }
    Ok(result)
}

struct Diagnostic {
    status: StatusCode,
    request_id: Option<String>,
}

impl Diagnostic {
    fn from_response(response: &Response) -> Self {
        Self {
            status: response.status(),
            request_id: response
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| diagnostic_field(value, b"_-"))
                .map(str::to_owned),
        }
    }

    fn failure(&self, summary: &str, error: Option<&Value>) -> LlmError {
        let mut message = format!(
            "ChatGPT subscription: {summary} (HTTP {}",
            self.status.as_u16()
        );
        let field = |name, extra| {
            error
                .and_then(|value| value.get(name))
                .and_then(Value::as_str)
                .and_then(|value| diagnostic_field(value, extra))
        };
        let code = field("code", b"_-");
        if let Some(code) = code {
            message.push_str(&format!(", code={code}"));
        }
        if let Some(param) = field("param", b"_-.[]") {
            message.push_str(&format!(", param={param}"));
        }
        if let Some(id) = &self.request_id {
            message.push_str(&format!(", request_id={id}"));
        }
        message.push(')');
        let guidance = match code {
            Some("model_not_found") => {
                "; choose a slug from the active account: edgecrab auth models chatgpt-pro"
            }
            Some("subscription_sharing_user_not_eligible") => {
                "; plan usage is not available for this account, workspace, or policy"
            }
            Some("subscription_sharing_usage_limit_exceeded") => {
                "; pause plan requests and check https://chatgpt.com/settings/usage"
            }
            Some(
                "subscription_sharing_usage_unavailable" | "subscription_sharing_user_unavailable",
            ) => "; keep the selected account and try again later",
            Some("subscription_sharing_unsupported_capability") => {
                "; remove the unsupported capability before retrying"
            }
            Some("subscription_sharing_route_not_supported") => {
                "; verify the SIWC method and endpoint"
            }
            Some("subscription_sharing_invalid_user") => {
                "; the subscriber context could not be validated"
            }
            Some(
                "chatpass_v2_scope_not_authorized" | "chatpass_v2_invalid_authorization_context",
            ) => "; verify the OAuth client and plan permission",
            _ => "",
        };
        message.push_str(guidance);
        // Don't classify subscription limits as transient rate limits or attach raw bodies.
        LlmError::ApiError(message)
    }
}

fn diagnostic_field<'a>(value: &'a str, extra: &[u8]) -> Option<&'a str> {
    (!value.is_empty()
        && value.len() <= 128
        && !value.starts_with("sk-")
        && !value.starts_with("eyJ")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || extra.contains(&b)))
    .then_some(value)
}

fn nonempty(value: &Value) -> Option<&str> {
    value.as_str().filter(|s| !s.trim().is_empty())
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest(format!("ChatGPT subscription: {message}"))
}

struct InferenceRequest {
    body: Value,
    callable: HashSet<String>,
}

fn build_request(
    model: &str,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    choice: Option<ToolChoice>,
    options: Option<&CompletionOptions>,
) -> Result<InferenceRequest> {
    let mut input = Vec::new();
    if let Some(prompt) = options.and_then(|opts| opts.system_prompt.as_deref()) {
        input.push(json!({"type":"message", "role":"developer", "content":prompt}));
    }
    for message in messages {
        append_message(&mut input, message)?;
    }
    let mut body = json!({"model":model, "input":input, "store":false, "stream":true});
    let mut functions = Vec::with_capacity(tools.len());
    let mut declared = HashSet::new();
    for tool in tools {
        if tool.tool_type != "function" {
            return Err(invalid(
                "only local function tool definitions are supported",
            ));
        }
        let f = &tool.function;
        if f.name.is_empty() || !f.parameters.is_object() || !declared.insert(f.name.clone()) {
            return Err(invalid("invalid or duplicate function definition"));
        }
        // ToolDefinition::function defaults to strict=true even for schemas
        // with optional fields. Keep those fields optional and let EdgeCrab's
        // local tool registry validate arguments before dispatch.
        functions.push(json!({
            "type":"function", "name":f.name, "description":f.description,
            "parameters":f.parameters, "strict":false
        }));
    }
    let mut callable = declared;
    let mode = match choice {
        None => None,
        Some(ToolChoice::Auto(mode) | ToolChoice::Required(mode)) => match mode.as_str() {
            "none" => {
                callable.clear();
                Some("none")
            }
            "auto" => Some("auto"),
            "required" if !functions.is_empty() => Some("required"),
            _ => return Err(invalid("invalid tool choice or no tools available")),
        },
        Some(ToolChoice::Function {
            choice_type,
            function,
        }) => {
            if choice_type != "function" || !callable.contains(&function.name) {
                return Err(invalid("selected function is not available"));
            }
            // The flat forced-function selector has no namespace field. Require
            // exactly one call from a namespace containing only that function.
            functions.retain(|item| item["name"].as_str() == Some(&function.name));
            callable.retain(|name| name == &function.name);
            body["parallel_tool_calls"] = json!(false);
            Some("required")
        }
    };
    if !functions.is_empty() {
        body["tools"] = json!([{
            "type":"namespace", "name":NAMESPACE, "description":"Local EdgeCrab tools",
            "tools":functions
        }]);
        if let Some(mode) = mode {
            body["tool_choice"] = json!(mode);
        }
    }
    if let Some(options) = options {
        // Generic sampling, output limits, stop sequences and foreign-provider
        // options aren't part of this SIWC HTTP contract.
        if let Some(effort) = &options.reasoning_effort {
            body["reasoning"] = json!({"effort":effort});
        }
        match options.response_format.as_deref() {
            None | Some("text") => {}
            Some("json_object") => body["text"] = json!({"format":{"type":"json_object"}}),
            Some("json_schema") => {
                let schema = options
                    .response_schema
                    .as_ref()
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        invalid("JSON schema response format requires a schema object")
                    })?;
                body["text"] = json!({"format":{
                    "type":"json_schema", "name":"response", "schema":schema, "strict":true
                }});
            }
            Some(_) => return Err(invalid("unsupported response format")),
        }
    }
    Ok(InferenceRequest { body, callable })
}

fn append_message(input: &mut Vec<Value>, message: &ChatMessage) -> Result<()> {
    if message.has_images() && message.role != ChatRole::User {
        return Err(invalid("images are supported only in user messages"));
    }
    match message.role {
        ChatRole::System | ChatRole::User | ChatRole::Assistant => {
            let role = match message.role {
                ChatRole::System => "developer",
                role => role.as_str(),
            };
            let mut content = json!(message.content);
            if message.has_images() {
                let mut parts = Vec::new();
                if !message.content.is_empty() {
                    parts.push(json!({"type":"input_text", "text":message.content}));
                }
                for image in message.images.iter().flatten() {
                    if !image.is_supported_mime() || image.data.is_empty() {
                        return Err(invalid("unsupported or empty image input"));
                    }
                    parts.push(json!({
                        "type":"input_image", "image_url":image.to_api_url(),
                        "detail":image.detail.as_deref().unwrap_or("auto")
                    }));
                }
                content = json!(parts);
            }
            if message.role != ChatRole::Assistant
                || !message.content.is_empty()
                || message.tool_calls.as_ref().is_none_or(Vec::is_empty)
            {
                input.push(json!({"type":"message", "role":role, "content":content}));
            }
            for call in message.tool_calls.iter().flatten() {
                if message.role != ChatRole::Assistant
                    || call.call_type != "function"
                    || call.id.is_empty()
                    || call.name().is_empty()
                {
                    return Err(invalid("invalid historical function call"));
                }
                input.push(json!({
                    "type":"function_call", "call_id":call.id, "namespace":NAMESPACE,
                    "name":call.name(), "arguments":call.arguments()
                }));
            }
        }
        ChatRole::Tool | ChatRole::Function => {
            let call_id = message
                .tool_call_id
                .as_deref()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| invalid("function output is missing its call ID"))?;
            input.push(json!({
                "type":"function_call_output", "call_id":call_id, "output":message.content
            }));
        }
    }
    Ok(())
}

// A byte framer keeps split UTF-8 intact and accepts LF, CRLF and CR. It only
// dispatches blank-line-terminated events, never an unterminated frame at EOF.
#[derive(Default)]
struct Sse {
    line: Vec<u8>,
    data: Vec<u8>,
    name: Option<Vec<u8>>,
    skip_lf: bool,
}

struct Frame {
    name: Option<Vec<u8>>,
    data: Vec<u8>,
}

impl Sse {
    fn push(&mut self, byte: u8) -> std::result::Result<Option<Frame>, ()> {
        if std::mem::take(&mut self.skip_lf) && byte == b'\n' {
            return Ok(None);
        }
        if byte != b'\n' && byte != b'\r' {
            if self.line.len() + self.data.len() >= MAX_EVENT_BYTES {
                return Err(());
            }
            self.line.push(byte);
            return Ok(None);
        }
        self.skip_lf = byte == b'\r';
        if self.line.is_empty() {
            let name = self.name.take();
            if self.data.is_empty() {
                return Ok(None);
            }
            self.data.pop(); // SSE joins data lines with LF, then strips the last LF.
            return Ok(Some(Frame {
                name,
                data: std::mem::take(&mut self.data),
            }));
        }
        let colon = self
            .line
            .iter()
            .position(|&byte| byte == b':')
            .unwrap_or(self.line.len());
        let value = self.line.get(colon + 1..).unwrap_or_default();
        let value = value.strip_prefix(b" ").unwrap_or(value);
        match &self.line[..colon] {
            b"data" => {
                self.data.extend_from_slice(value);
                self.data.push(b'\n');
            }
            b"event" => self.name = Some(value.to_vec()),
            _ => {}
        }
        self.line.clear();
        Ok(None)
    }
}

enum Event {
    Text(String),
    Completed {
        response: LLMResponse,
        usage: Option<StreamUsage>,
        remaining_text: String,
    },
}

struct Decoder {
    sse: Sse,
    diagnostic: Diagnostic,
    model: String,
    callable: HashSet<String>,
    text: String,
}

impl Decoder {
    fn event(&mut self, name: Option<&[u8]>, data: &[u8]) -> Result<Option<Event>> {
        let failure = |summary| self.diagnostic.failure(summary, None);
        if data == b"[DONE]" {
            return Err(failure("stream ended without response.completed"));
        }
        let event: Value =
            serde_json::from_slice(data).map_err(|_| failure("malformed SSE event"))?;
        let event_type = event["type"]
            .as_str()
            .or_else(|| (name == Some(b"error")).then_some("error"));
        let event_type = event_type.ok_or_else(|| failure("SSE event missing its type"))?;
        if name.is_some_and(|name| {
            !name.is_empty() && name != b"message" && name != event_type.as_bytes()
        }) {
            return Err(failure("mismatched SSE event type"));
        }
        let response = &event["response"];
        match event_type {
            "response.output_text.delta" => {
                let delta = event["delta"]
                    .as_str()
                    .ok_or_else(|| failure("malformed text delta"))?;
                if self.text.len() + delta.len() > MAX_EVENT_BYTES {
                    return Err(failure("streamed text exceeds the size limit"));
                }
                self.text.push_str(delta);
                Ok(Some(Event::Text(delta.to_owned())))
            }
            "response.completed" => {
                let (response, usage) =
                    completed_response(response, &self.model, &self.callable)
                        .ok_or_else(|| failure("invalid or unsupported completed response"))?;
                let visible = if response.content.is_empty() {
                    response.refusal.as_deref().unwrap_or_default()
                } else {
                    &response.content
                };
                let remaining_text = visible
                    .strip_prefix(&self.text)
                    .ok_or_else(|| failure("completed response did not match streamed text"))?
                    .to_owned();
                Ok(Some(Event::Completed {
                    response,
                    usage,
                    remaining_text,
                }))
            }
            "response.failed" => Err(self
                .diagnostic
                .failure("response failed", response.get("error"))),
            "response.incomplete" => {
                let reason = response
                    .pointer("/incomplete_details/reason")
                    .and_then(Value::as_str);
                let summary = match reason {
                    Some("max_output_tokens") => "response incomplete (max_output_tokens)",
                    Some("max_messages") => "response incomplete (max_messages)",
                    Some("content_filter") => "response incomplete (content_filter)",
                    _ => "response incomplete",
                };
                Err(self.diagnostic.failure(summary, response.get("error")))
            }
            "response.cancelled" => Err(failure("response cancelled")),
            "error" => Err(self.diagnostic.failure(
                "stream returned an error",
                event.get("error").or(Some(&event)),
            )),
            _ => Ok(None), // Nonterminal lifecycle, argument, refusal and annotation events.
        }
    }
}

fn decode_stream<B, E>(
    body: impl Stream<Item = std::result::Result<B, E>> + Send + Unpin + 'static,
    diagnostic: Diagnostic,
    model: String,
    callable: HashSet<String>,
) -> BoxStream<'static, Result<Event>>
where
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
{
    let decoder = Decoder {
        sse: Sse::default(),
        diagnostic,
        model,
        callable,
        text: String::new(),
    };
    stream::try_unfold(
        (body, decoder, None::<B>, 0, false),
        |(mut body, mut decoder, mut chunk, mut offset, finished)| async move {
            if finished {
                return Ok(None);
            }
            loop {
                if let Some(bytes) = &chunk {
                    while let Some(&byte) = bytes.as_ref().get(offset) {
                        offset += 1;
                        let frame = decoder.sse.push(byte).map_err(|()| {
                            decoder
                                .diagnostic
                                .failure("SSE event exceeds the size limit", None)
                        })?;
                        if let Some(Frame { name, data }) = frame
                            && let Some(event) = decoder.event(name.as_deref(), &data)?
                        {
                            let finished = matches!(event, Event::Completed { .. });
                            return Ok(Some((event, (body, decoder, chunk, offset, finished))));
                        }
                    }
                }
                chunk = match body.next().await {
                    Some(Ok(bytes)) => Some(bytes),
                    Some(Err(_)) => {
                        return Err(decoder.diagnostic.failure("stream interrupted", None));
                    }
                    None => {
                        return Err(decoder
                            .diagnostic
                            .failure("stream ended without response.completed", None));
                    }
                };
                offset = 0;
            }
        },
    )
    .boxed()
}

fn completed_response(
    value: &Value,
    model: &str,
    callable: &HashSet<String>,
) -> Option<(LLMResponse, Option<StreamUsage>)> {
    if value["status"].as_str() != Some("completed") || !value["error"].is_null() {
        return None;
    }
    let mut response = LLMResponse::new("", value["model"].as_str().unwrap_or(model));
    let mut call_ids = HashSet::new();
    for item in value["output"].as_array()? {
        if !item["status"].is_null() && item["status"].as_str() != Some("completed") {
            return None;
        }
        match item["type"].as_str()? {
            "message" => append_output_message(&mut response, item)?,
            "function_call" => {
                let name = nonempty(&item["name"])?;
                let id = nonempty(&item["call_id"])?;
                if (!item["namespace"].is_null() && item["namespace"].as_str() != Some(NAMESPACE))
                    || !callable.contains(name)
                    || !call_ids.insert(id)
                {
                    return None;
                }
                response.tool_calls.push(ToolCall {
                    id: id.to_owned(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: name.to_owned(),
                        arguments: item["arguments"].as_str()?.to_owned(),
                    },
                    thought_signature: None,
                });
            }
            // ChatMessage has no Responses reasoning item slot. We replay all
            // visible messages and calls, not provider-specific hidden reasoning.
            "reasoning" => {}
            _ => return None,
        }
    }
    if response.content.is_empty() && response.refusal.is_none() && response.tool_calls.is_empty() {
        return None;
    }
    response.finish_reason = Some(
        if response.tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        }
        .into(),
    );
    let usage = if value["usage"].is_null() {
        None
    } else {
        Some(apply_usage(&mut response, &value["usage"])?)
    };
    Some((response, usage))
}

fn append_output_message(response: &mut LLMResponse, item: &Value) -> Option<()> {
    if item["role"].as_str() != Some("assistant") {
        return None;
    }
    for part in item["content"].as_array()? {
        match part["type"].as_str()? {
            "output_text" => response.content.push_str(part["text"].as_str()?),
            "refusal" => response
                .refusal
                .get_or_insert_with(String::new)
                .push_str(part["refusal"].as_str()?),
            _ => return None,
        }
    }
    Some(())
}

fn apply_usage(response: &mut LLMResponse, usage: &Value) -> Option<StreamUsage> {
    let count = |value: &Value| value.as_u64().and_then(|n| usize::try_from(n).ok());
    let mut snapshot = StreamUsage::new(
        count(&usage["input_tokens"])?,
        count(&usage["output_tokens"])?,
    );
    snapshot.cache_hit_tokens = count(&usage["input_tokens_details"]["cached_tokens"]);
    snapshot.thinking_tokens = count(&usage["output_tokens_details"]["reasoning_tokens"]);
    response.prompt_tokens = snapshot.prompt_tokens;
    response.completion_tokens = snapshot.completion_tokens;
    response.total_tokens = count(&usage["total_tokens"]).unwrap_or_else(|| {
        snapshot
            .prompt_tokens
            .saturating_add(snapshot.completion_tokens)
    });
    response.cache_hit_tokens = snapshot.cache_hit_tokens;
    response.thinking_tokens = snapshot.thinking_tokens;
    Some(snapshot)
}

fn to_chunks(events: BoxStream<'static, Result<Event>>) -> BoxStream<'static, Result<StreamChunk>> {
    events
        .flat_map(|event| {
            let chunks = match event {
                Err(error) => vec![Err(error)],
                Ok(Event::Text(text)) => vec![Ok(StreamChunk::Content(text))],
                Ok(Event::Completed {
                    response,
                    usage,
                    remaining_text,
                }) => {
                    let mut chunks = Vec::new();
                    if !remaining_text.is_empty() {
                        chunks.push(Ok(StreamChunk::Content(remaining_text)));
                    }
                    // Only publish executable tool calls from the validated terminal
                    // snapshot, never from an interrupted/incomplete response draft.
                    for (index, call) in response.tool_calls.into_iter().enumerate() {
                        chunks.push(Ok(StreamChunk::ToolCallDelta {
                            index,
                            id: Some(call.id),
                            function_name: Some(call.function.name),
                            function_arguments: Some(call.function.arguments),
                            thought_signature: None,
                        }));
                    }
                    chunks.push(Ok(StreamChunk::Finished {
                        reason: response.finish_reason.unwrap_or_else(|| "stop".into()),
                        ttft_ms: None,
                        usage,
                    }));
                    chunks
                }
            };
            stream::iter(chunks)
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgequake_llm::traits::ImageData;

    fn call() -> ToolCall {
        ToolCall {
            id: "call_123".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: r#"{"path":"a"}"#.into(),
            },
            thought_signature: None,
        }
    }

    fn schema() -> Value {
        json!({"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"], "additionalProperties":false})
    }

    #[test]
    fn stateless_request_replays_history_and_only_supported_fields() {
        let messages = [
            ChatMessage::system("system instruction"),
            ChatMessage::user_with_images("first", vec![ImageData::new("aW1hZ2U=", "image/png")]),
            ChatMessage::assistant_with_tools("I'll read it", vec![call()]),
            ChatMessage::tool_result("call_123", "file content"),
            ChatMessage::user("next"),
            ChatMessage::system("mid-history summary"),
        ];
        let options = CompletionOptions {
            system_prompt: Some("options instruction".into()),
            max_tokens: Some(100),
            temperature: Some(0.5),
            top_p: Some(0.9),
            stop: Some(vec!["stop".into()]),
            reasoning_effort: Some("low".into()),
            response_format: Some("json_schema".into()),
            response_schema: Some(schema()),
            ..Default::default()
        };
        let tool = ToolDefinition::function("read_file", "Read a file", schema());
        assert_eq!(tool.function.strict, Some(true));
        let request = build_request(
            "model-slug",
            &messages,
            &[tool],
            Some(ToolChoice::auto()),
            Some(&options),
        )
        .unwrap();
        assert_eq!(
            request.body,
            json!({
                "model":"model-slug", "store":false, "stream":true,
                "input":[
                    {"type":"message", "role":"developer", "content":"options instruction"},
                    {"type":"message", "role":"developer", "content":"system instruction"},
                    {"type":"message", "role":"user", "content":[
                        {"type":"input_text", "text":"first"},
                        {"type":"input_image", "image_url":"data:image/png;base64,aW1hZ2U=", "detail":"auto"}
                    ]},
                    {"type":"message", "role":"assistant", "content":"I'll read it"},
                    {"type":"function_call", "call_id":"call_123", "namespace":"edgecrab", "name":"read_file", "arguments":r#"{"path":"a"}"#},
                    {"type":"function_call_output", "call_id":"call_123", "output":"file content"},
                    {"type":"message", "role":"user", "content":"next"},
                    {"type":"message", "role":"developer", "content":"mid-history summary"}
                ],
                "tools":[{"type":"namespace", "name":"edgecrab", "description":"Local EdgeCrab tools", "tools":[
                    {"type":"function", "name":"read_file", "description":"Read a file", "parameters":schema(), "strict":false}
                ]}],
                "tool_choice":"auto", "reasoning":{"effort":"low"},
                "text":{"format":{"type":"json_schema", "name":"response", "schema":schema(), "strict":true}}
            })
        );
    }

    #[test]
    fn forced_tool_stays_in_namespace_and_unsupported_tools_error() {
        let mut tools = [
            ToolDefinition::function("read_file", "Read", schema()),
            ToolDefinition::function("write_file", "Write", schema()),
        ];
        let request = build_request(
            "m",
            &[],
            &tools,
            Some(ToolChoice::function("read_file")),
            None,
        )
        .unwrap();
        assert_eq!(request.body["tool_choice"], "required");
        assert_eq!(request.body["parallel_tool_calls"], false);
        let nested = request.body["tools"][0]["tools"].as_array().unwrap();
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0]["name"], "read_file");
        assert_eq!(request.callable, HashSet::from(["read_file".into()]));
        tools[1].tool_type = "file_search".into();
        assert!(build_request("m", &[], &tools, None, None).is_err());
        assert!(
            build_request(
                "m",
                &[ChatMessage::tool_result("", "secret")],
                &[],
                None,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn event_stream_may_omit_the_content_type() {
        use header::HeaderValue;
        assert!(stream_content_type_supported(None));
        assert!(stream_content_type_supported(Some(
            &HeaderValue::from_static("text/event-stream; charset=utf-8")
        )));
        assert!(!stream_content_type_supported(Some(
            &HeaderValue::from_static("application/json")
        )));
    }

    #[test]
    fn catalog_preserves_display_order_and_rejects_wrong_shape() {
        let models = parse_models(
            br#"{"models":[
            {"slug":"z","display_name":"Zed","visibility":"list"},
            {"visibility":"hidden"},
            {"slug":"a","display_name":"Alpha","visibility":"list"}
        ]}"#,
        )
        .unwrap();
        assert_eq!(
            models,
            [
                SubscriptionModel {
                    id: "z".into(),
                    label: "Zed".into()
                },
                SubscriptionModel {
                    id: "a".into(),
                    label: "Alpha".into()
                }
            ]
        );
        assert!(parse_models(br#"{"models":[]}"#).unwrap().is_empty());
        assert!(parse_models(br#"{"data":[]}"#).is_err());
        assert!(parse_models(br#"{"models":[{"visibility":"list","slug":"a"}]}"#).is_err());
    }

    fn diagnostic(status: StatusCode) -> Diagnostic {
        Diagnostic {
            status,
            request_id: Some("req_test".into()),
        }
    }

    fn event_stream(bytes: &str) -> BoxStream<'static, Result<Event>> {
        // One byte per chunk exercises arbitrary boundary placement, including UTF-8.
        let bytes: Vec<_> = bytes.bytes().map(|byte| Ok::<_, ()>([byte])).collect();
        decode_stream(
            stream::iter(bytes),
            diagnostic(StatusCode::OK),
            "requested".into(),
            HashSet::from(["read_file".into()]),
        )
    }

    fn completion() -> Value {
        json!({"type":"response.completed", "response":{
            "status":"completed", "model":"served", "output":[
                {"type":"reasoning"},
                {"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"é ok"}]},
                {"type":"function_call", "id":"fc_item", "call_id":"call_123", "name":"read_file", "namespace":"edgecrab", "arguments":r#"{"path":"a"}"#}
            ],
            "usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13,
                "input_tokens_details":{"cached_tokens":5},"output_tokens_details":{"reasoning_tokens":2}}
        }})
    }

    #[tokio::test]
    async fn fragmented_sse_uses_completed_snapshot_and_finishes_without_waiting_for_eof() {
        let fixture = format!(
            concat!(
                ": heartbeat\r\n\r\n",
                "event: response.output_text.delta\r\n",
                "data: {{\"type\":\"response.output_text.delta\",\r\n",
                "data: \"delta\":\"é\"}}\r\n\r\n",
                "data: {{\"type\":\"response.function_call_arguments.delta\",\"delta\":\"draft\"}}\n\n",
                "data: {}\n\n",
                "data: not JSON\n\n",
            ),
            completion()
        );
        let chunks = to_chunks(event_stream(&fixture)).collect::<Vec<_>>().await;
        assert!(matches!(&chunks[..], [
            Ok(StreamChunk::Content(first)),
            Ok(StreamChunk::Content(rest)),
            Ok(StreamChunk::ToolCallDelta {index:0, id:Some(id), function_name:Some(name), function_arguments:Some(args), ..}),
            Ok(StreamChunk::Finished {reason, usage:Some(usage), ..}),
        ] if first == "é" && rest == " ok" && id == "call_123" && name == "read_file" && args == call().arguments()
            && reason == "tool_calls" && usage.prompt_tokens == 9 && usage.completion_tokens == 4
            && usage.cache_hit_tokens == Some(5) && usage.thinking_tokens == Some(2)));
        let events = event_stream(&fixture).collect::<Vec<_>>().await;
        assert!(
            matches!(&events[..], [Ok(Event::Text(_)), Ok(Event::Completed {response, ..})]
            if response.model == "served" && response.content == "é ok" && response.total_tokens == 13)
        );
    }

    #[tokio::test]
    async fn unsuccessful_streams_never_emit_finished_or_executable_calls() {
        let draft = "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\"}}\n\n";
        let mut foreign = completion();
        foreign["response"]["output"][2]["namespace"] = json!("foreign");
        for (tail, expected) in [
            (String::new(), "without response.completed"),
            ("data: [DONE]\n\n".into(), "without response.completed"),
            (format!("data: {}\n", completion()), "without response.completed"),
            ("data: malformed secret\n\n".into(), "malformed SSE event"),
            ("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n".into(), "incomplete (content_filter)"),
            ("data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"subscription_sharing_usage_limit_exceeded\",\"message\":\"secret\"}}}\n\n".into(), "code=subscription_sharing_usage_limit_exceeded"),
            ("event: error\ndata: {\"error\":{\"code\":\"server_error\",\"message\":\"secret\"}}\n\n".into(), "code=server_error"),
            (format!("data: {foreign}\n\n"), "invalid or unsupported completed response"),
        ] {
            let chunks = to_chunks(event_stream(&format!("{draft}{tail}"))).collect::<Vec<_>>().await;
            let [Err(error)] = &chunks[..] else { panic!("expected one terminal error") };
            let message = error.to_string();
            assert!(message.contains(expected));
            assert!(message.contains("HTTP 200, ") && message.contains("request_id=req_test"));
            assert!(!message.contains("secret"));
        }
    }

    #[tokio::test]
    async fn body_failure_after_text_is_a_terminal_error() {
        let body = stream::iter([
            Ok(&b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"[..]),
            Err("raw transport error"),
        ]);
        let events = decode_stream(
            body,
            diagnostic(StatusCode::OK),
            "requested".into(),
            HashSet::new(),
        );
        let chunks = to_chunks(events).collect::<Vec<_>>().await;
        assert!(
            matches!(&chunks[..], [Ok(StreamChunk::Content(text)), Err(error)]
            if text == "partial" && error.to_string().contains("stream interrupted")
                && !error.to_string().contains("raw transport error"))
        );
    }

    #[test]
    fn unavailable_model_points_to_account_catalog() {
        let error = json!({"code":"model_not_found"});
        let rendered = diagnostic(StatusCode::BAD_REQUEST)
            .failure("HTTP request rejected", Some(&error))
            .to_string();
        assert!(rendered.contains("code=model_not_found"));
        assert!(rendered.contains("edgecrab auth models chatgpt-pro"));
    }

    #[test]
    fn diagnostic_keeps_status_and_safe_fields_not_raw_error() {
        let error = json!({"code":"subscription_sharing_unsupported_capability", "param":"tools[0].type", "message":"private detail"});
        let rendered = diagnostic(StatusCode::BAD_REQUEST)
            .failure("HTTP request rejected", Some(&error))
            .to_string();
        assert!(rendered.contains("HTTP 400, code=subscription_sharing_unsupported_capability, param=tools[0].type, request_id=req_test"));
        assert!(!rendered.contains("private detail"));
        let malicious = json!({"code":"sk-this-is-a-secret", "param":"\nforged log"});
        let err = diagnostic(StatusCode::FORBIDDEN).failure("request forbidden", Some(&malicious));
        assert!(!format!("{err:?}").contains("secret"));
        assert!(!err.to_string().contains("forged"));
    }
}
